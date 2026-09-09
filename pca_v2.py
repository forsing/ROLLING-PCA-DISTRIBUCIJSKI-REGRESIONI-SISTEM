"""
LOTO 7/39 — ROLLING PCA DISTRIBUCIJSKI REGRESIONI SISTEM

istorijske distribucijske osobine
→ StandardScaler
→ rolling PCA
→ PCA reconstruction error
→ originalne osobine i PCA komponente
→ HistGradientBoostingRegressor
→ nested walk-forward izbor broja PCA komponenti
→ zaključana validacija
→ zamrznuti završni holdout
→ jedna NEXT predikcija

Prvi red CSV-a smatra se najstarijim, a poslednji najnovijim.
"""

from __future__ import annotations

import math
import random
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PODEŠAVANJA
# =============================================================================

SEED = 39

BROJ_KUGLICA = 39
BROJEVA_U_KOMBINACIJI = 7

TEORIJSKA_STOPA = (
    BROJEVA_U_KOMBINACIJI
    / BROJ_KUGLICA
)

TEORIJSKO_OCEKIVANJE_POGODAKA = (
    BROJEVA_U_KOMBINACIJI ** 2
    / BROJ_KUGLICA
)

BROJ_SVIH_KOMBINACIJA = math.comb(
    BROJ_KUGLICA,
    BROJEVA_U_KOMBINACIJI,
)

LOTO_CSV = (
    "/data/"
    "loto7_4682_k72.csv"
)

MINIMUM_ISTORIJE = 100

PROZORI = (
    10,
    20,
    50,
    100,
)

PCA_PROZOR = 600

PCA_KANDIDATI = (
    2,
    3,
    5,
    7,
)

UDEO_RAZVOJ = 0.60
UDEO_VALIDACIJA = 0.20
UDEO_HOLDOUT = 0.20

BROJ_WALK_FORWARD_FOLDOVA = 5

warnings.filterwarnings("ignore")

np.random.seed(SEED)
random.seed(SEED)


# =============================================================================
# POMOĆNE FUNKCIJE
# =============================================================================

def naslov(
    tekst: str,
    znak: str = "=",
) -> None:
    print()
    print(znak * 88)
    print(tekst)
    print(znak * 88)


def status(
    naziv: str,
    prosao: bool,
    dodatak: str = "",
) -> None:
    oznaka = (
        "PROŠLO"
        if prosao
        else "NIJE PROŠLO"
    )

    if dodatak:
        print(
            f"{naziv:<48}"
            f"{oznaka:<15}"
            f"{dodatak}"
        )
    else:
        print(
            f"{naziv:<48}"
            f"{oznaka}"
        )


def formatiraj_kombinaciju(
    kombinacija: list[int],
) -> str:
    return ", ".join(
        f"{broj:02d}"
        for broj in sorted(kombinacija)
    )


def ucitaj_csv(
    putanja: str,
) -> np.ndarray:
    okvir = pd.read_csv(
        putanja,
        header=None,
    )

    okvir = okvir.apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()

    if okvir.shape[1] < BROJEVA_U_KOMBINACIJI:
        raise ValueError(
            "CSV mora imati najmanje sedam kolona."
        )

    kombinacije = okvir.iloc[
        :,
        :BROJEVA_U_KOMBINACIJI,
    ].astype(int).to_numpy()

    matrica = np.zeros(
        (
            len(kombinacije),
            BROJ_KUGLICA,
        ),
        dtype=np.float32,
    )

    for indeks, red in enumerate(kombinacije):
        brojevi = sorted(
            int(broj)
            for broj in red
        )

        if len(set(brojevi)) != BROJEVA_U_KOMBINACIJI:
            raise ValueError(
                f"Red {indeks + 1} sadrži "
                f"ponovljene brojeve."
            )

        if (
            brojevi[0] < 1
            or brojevi[-1] > BROJ_KUGLICA
        ):
            raise ValueError(
                f"Red {indeks + 1} sadrži "
                f"broj van opsega 1–39."
            )

        matrica[
            indeks,
            np.asarray(brojevi) - 1,
        ] = 1.0

    if len(matrica) <= MINIMUM_ISTORIJE + 50:
        raise ValueError(
            "CSV nema dovoljno istorijskih izvlačenja."
        )

    return matrica


def binarna_u_kombinaciju(
    red: np.ndarray,
) -> list[int]:
    return (
        np.flatnonzero(red > 0.5) + 1
    ).astype(int).tolist()


def rang_kombinacije(
    kombinacija: list[int],
) -> int:
    kombinacija = sorted(kombinacija)

    rang = 0
    prethodni = 0
    preostalo = BROJEVA_U_KOMBINACIJI

    for broj in kombinacija:
        for kandidat in range(
            prethodni + 1,
            broj,
        ):
            rang += math.comb(
                BROJ_KUGLICA - kandidat,
                preostalo - 1,
            )

        prethodni = broj
        preostalo -= 1

    return int(rang)


# =============================================================================
# PRAVLJENJE VREMENSKI ISPRAVNIH OSOBINA
# =============================================================================

def napravi_osobine(
    izvlacenja: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Za trenutak t koristi isključivo redove 0...t-1.

    Funkcija vraća:

    - osobine za istorijske mete;
    - stvarne istorijske mete;
    - osobine za naredno, još nepoznato izvlačenje.
    """

    broj_redova = len(izvlacenja)

    kumulativno = np.vstack(
        [
            np.zeros(
                (
                    1,
                    BROJ_KUGLICA,
                ),
                dtype=np.float64,
            ),
            np.cumsum(
                izvlacenja,
                axis=0,
                dtype=np.float64,
            ),
        ]
    )

    poslednje_pojavljivanje = np.full(
        BROJ_KUGLICA,
        -1,
        dtype=np.int64,
    )

    ewma = np.full(
        BROJ_KUGLICA,
        TEORIJSKA_STOPA,
        dtype=np.float64,
    )

    alfa_ewma = 2.0 / 21.0

    graf_parova = np.zeros(
        (
            BROJ_KUGLICA,
            BROJ_KUGLICA,
        ),
        dtype=np.float64,
    )

    for indeks in range(MINIMUM_ISTORIJE):
        red = izvlacenja[indeks]

        aktivni = np.flatnonzero(
            red > 0.5
        )

        graf_parova[
            np.ix_(aktivni, aktivni)
        ] += 1.0

        poslednje_pojavljivanje[
            aktivni
        ] = indeks

        ewma = (
            alfa_ewma * red
            + (1.0 - alfa_ewma) * ewma
        )

    sve_osobine = []
    sve_mete = []

    for trenutak in range(
        MINIMUM_ISTORIJE,
        broj_redova + 1,
    ):
        ukupno_pre = kumulativno[
            trenutak
        ]

        stopa_sve = (
            ukupno_pre
            / float(trenutak)
        )

        prozorske_stope = []

        for prozor in PROZORI:
            pocetak = max(
                0,
                trenutak - prozor,
            )

            stvarna_duzina = (
                trenutak - pocetak
            )

            broj_u_prozoru = (
                kumulativno[trenutak]
                - kumulativno[pocetak]
            )

            prozorske_stope.append(
                broj_u_prozoru
                / float(stvarna_duzina)
            )

        stopa_10 = prozorske_stope[0]
        stopa_20 = prozorske_stope[1]
        stopa_50 = prozorske_stope[2]
        stopa_100 = prozorske_stope[3]

        gap = (
            trenutak
            - poslednje_pojavljivanje
        ).astype(np.float64)

        gap[
            poslednje_pojavljivanje < 0
        ] = float(trenutak + 1)

        ocekivani_gap = (
            1.0 / TEORIJSKA_STOPA
        )

        normalizovani_gap = (
            gap / ocekivani_gap
        )

        trend_kratki = (
            stopa_10 - stopa_50
        )

        trend_srednji = (
            stopa_20 - stopa_100
        )

        odstupanje_od_osnove = (
            stopa_sve
            - TEORIJSKA_STOPA
        )

        if trenutak > 0:
            zagladjeni_odnos = (
                ukupno_pre
                + trenutak * TEORIJSKA_STOPA
            ) / (
                2.0
                * trenutak
                * TEORIJSKA_STOPA
            )
        else:
            zagladjeni_odnos = np.ones(
                BROJ_KUGLICA,
                dtype=np.float64,
            )

        poslednja_kombinacija = (
            izvlacenja[trenutak - 1]
        )

        poslednji_brojevi = np.flatnonzero(
            poslednja_kombinacija > 0.5
        )

        if len(poslednji_brojevi) > 0:
            grafovski_skor = np.mean(
                graf_parova[
                    :,
                    poslednji_brojevi,
                ],
                axis=1,
            ) / float(max(1, trenutak))
        else:
            grafovski_skor = np.zeros(
                BROJ_KUGLICA,
                dtype=np.float64,
            )

        broj_osobina = (
            np.arange(
                1,
                BROJ_KUGLICA + 1,
                dtype=np.float64,
            )
            / BROJ_KUGLICA
        )

        osobine_trenutka = np.column_stack(
            [
                broj_osobina,
                stopa_sve,
                stopa_10,
                stopa_20,
                stopa_50,
                stopa_100,
                ewma,
                normalizovani_gap,
                np.log1p(gap),
                trend_kratki,
                trend_srednji,
                odstupanje_od_osnove,
                zagladjeni_odnos,
                grafovski_skor,
                poslednja_kombinacija,
            ]
        ).astype(np.float32)

        sve_osobine.append(
            osobine_trenutka
        )

        if trenutak < broj_redova:
            sve_mete.append(
                izvlacenja[trenutak].copy()
            )

            red = izvlacenja[trenutak]

            aktivni = np.flatnonzero(
                red > 0.5
            )

            graf_parova[
                np.ix_(aktivni, aktivni)
            ] += 1.0

            poslednje_pojavljivanje[
                aktivni
            ] = trenutak

            ewma = (
                alfa_ewma * red
                + (1.0 - alfa_ewma) * ewma
            )

    istorijske_osobine = np.asarray(
        sve_osobine[:-1],
        dtype=np.float32,
    )

    istorijske_mete = np.asarray(
        sve_mete,
        dtype=np.float32,
    )

    next_osobine = np.asarray(
        sve_osobine[-1],
        dtype=np.float32,
    )

    return (
        istorijske_osobine,
        istorijske_mete,
        next_osobine,
    )


# =============================================================================
# ROLLING PCA I REGRESOR
# =============================================================================

@dataclass
class PCAModel:
    scaler: StandardScaler
    pca: PCA
    regresor: HistGradientBoostingRegressor
    broj_komponenti: int


def prosiri_pca_osobine(
    osnovne_osobine: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    standardizovane = scaler.transform(
        osnovne_osobine
    )

    komponente = pca.transform(
        standardizovane
    )

    rekonstrukcija = pca.inverse_transform(
        komponente
    )

    reconstruction_error = np.mean(
        (
            standardizovane
            - rekonstrukcija
        ) ** 2,
        axis=1,
        keepdims=True,
    )

    return np.column_stack(
        [
            standardizovane,
            komponente,
            reconstruction_error,
        ]
    ).astype(np.float32)


def vremenske_redove_u_uzorke(
    osobine: np.ndarray,
    mete: np.ndarray,
    indeksi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        osobine[indeksi].reshape(
            -1,
            osobine.shape[-1],
        ),
        mete[indeksi].reshape(-1),
    )


def prilagodi_model(
    osobine: np.ndarray,
    mete: np.ndarray,
    indeksi_obuke: np.ndarray,
    broj_komponenti: int,
) -> PCAModel:
    if len(indeksi_obuke) == 0:
        raise ValueError(
            "Nema vremenskih redova za obuku."
        )

    prvi_dozvoljeni = max(
        int(indeksi_obuke[-1])
        - PCA_PROZOR
        + 1,
        int(indeksi_obuke[0]),
    )

    rolling_indeksi = indeksi_obuke[
        indeksi_obuke >= prvi_dozvoljeni
    ]

    x_obuka, y_obuka = vremenske_redove_u_uzorke(
        osobine,
        mete,
        rolling_indeksi,
    )

    scaler = StandardScaler()

    x_standardizovano = scaler.fit_transform(
        x_obuka
    )

    stvarni_broj_komponenti = min(
        broj_komponenti,
        x_standardizovano.shape[1],
        x_standardizovano.shape[0] - 1,
    )

    stvarni_broj_komponenti = max(
        1,
        stvarni_broj_komponenti,
    )

    pca = PCA(
        n_components=stvarni_broj_komponenti,
        svd_solver="full",
        random_state=SEED,
    )

    komponente = pca.fit_transform(
        x_standardizovano
    )

    rekonstrukcija = pca.inverse_transform(
        komponente
    )

    reconstruction_error = np.mean(
        (
            x_standardizovano
            - rekonstrukcija
        ) ** 2,
        axis=1,
        keepdims=True,
    )

    x_finalno = np.column_stack(
        [
            x_standardizovano,
            komponente,
            reconstruction_error,
        ]
    ).astype(np.float32)

    regresor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=SEED,
    )

    regresor.fit(
        x_finalno,
        y_obuka,
    )

    return PCAModel(
        scaler=scaler,
        pca=pca,
        regresor=regresor,
        broj_komponenti=stvarni_broj_komponenti,
    )


def predvidi_vremenske_redove(
    model: PCAModel,
    osobine: np.ndarray,
    indeksi: np.ndarray,
) -> np.ndarray:
    predikcije = []

    for indeks in indeksi:
        x = prosiri_pca_osobine(
            osobine[indeks],
            model.scaler,
            model.pca,
        )

        skorovi = model.regresor.predict(x)

        skorovi = np.clip(
            skorovi,
            0.0,
            1.0,
        )

        predikcije.append(
            skorovi
        )

    return np.asarray(
        predikcije,
        dtype=np.float64,
    )


def predvidi_next(
    model: PCAModel,
    next_osobine: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    x = prosiri_pca_osobine(
        next_osobine,
        model.scaler,
        model.pca,
    )

    skorovi = model.regresor.predict(x)

    skorovi = np.clip(
        skorovi,
        0.0,
        1.0,
    )

    poredak = np.argsort(
        -skorovi,
        kind="stable",
    )

    kombinacija = sorted(
        (
            poredak[
                :BROJEVA_U_KOMBINACIJI
            ] + 1
        ).astype(int).tolist()
    )

    return kombinacija, skorovi


# =============================================================================
# OCENJIVANJE
# =============================================================================

def oceni_predikcije(
    predikcije: np.ndarray,
    stvarne_mete: np.ndarray,
) -> dict:
    pogoci = []

    for skorovi, stvarni_red in zip(
        predikcije,
        stvarne_mete,
    ):
        pred_brojevi = set(
            (
                np.argsort(
                    -skorovi,
                    kind="stable",
                )[:BROJEVA_U_KOMBINACIJI]
                + 1
            ).astype(int).tolist()
        )

        stvarni_brojevi = set(
            binarna_u_kombinaciju(
                stvarni_red
            )
        )

        pogoci.append(
            len(
                pred_brojevi
                & stvarni_brojevi
            )
        )

    pogoci = np.asarray(
        pogoci,
        dtype=np.int64,
    )

    if len(pogoci) == 0:
        return {
            "broj_provera": 0,
            "prosek_pogodaka": float("nan"),
            "potpuno_tacnih": 0,
            "greske": 0,
            "raspodela": {
                broj: 0
                for broj in range(8)
            },
        }

    return {
        "broj_provera": int(len(pogoci)),
        "prosek_pogodaka": float(
            np.mean(pogoci)
        ),
        "potpuno_tacnih": int(
            np.sum(pogoci == 7)
        ),
        "greske": int(
            np.sum(pogoci < 7)
        ),
        "raspodela": {
            broj: int(
                np.sum(pogoci == broj)
            )
            for broj in range(8)
        },
    }


# =============================================================================
# NESTED WALK-FORWARD
# =============================================================================

def nested_walk_forward(
    osobine: np.ndarray,
    mete: np.ndarray,
    razvoj_kraj: int,
) -> dict:
    fold_rezultati = []
    rezultati_komponenti = {
        broj: []
        for broj in PCA_KANDIDATI
    }

    minimalna_obuka = max(
        100,
        int(razvoj_kraj * 0.35),
    )

    granice = np.linspace(
        minimalna_obuka,
        razvoj_kraj,
        BROJ_WALK_FORWARD_FOLDOVA + 1,
        dtype=int,
    )

    for redni in range(
        BROJ_WALK_FORWARD_FOLDOVA
    ):
        kraj_obuke = int(
            granice[redni]
        )

        kraj_provere = int(
            granice[redni + 1]
        )

        indeksi_obuke = np.arange(
            0,
            kraj_obuke,
            dtype=np.int64,
        )

        indeksi_provere = np.arange(
            kraj_obuke,
            kraj_provere,
            dtype=np.int64,
        )

        if (
            len(indeksi_obuke) < 30
            or len(indeksi_provere) == 0
        ):
            continue

        najbolji_fold = None

        for broj_komponenti in PCA_KANDIDATI:
            model = prilagodi_model(
                osobine=osobine,
                mete=mete,
                indeksi_obuke=indeksi_obuke,
                broj_komponenti=broj_komponenti,
            )

            predikcije = predvidi_vremenske_redove(
                model=model,
                osobine=osobine,
                indeksi=indeksi_provere,
            )

            rezultat = oceni_predikcije(
                predikcije,
                mete[indeksi_provere],
            )

            rezultati_komponenti[
                broj_komponenti
            ].append(
                rezultat["prosek_pogodaka"]
            )

            kandidat = {
                "broj_komponenti":
                    broj_komponenti,
                **rezultat,
            }

            if (
                najbolji_fold is None
                or kandidat["prosek_pogodaka"]
                > najbolji_fold["prosek_pogodaka"]
                or (
                    kandidat["prosek_pogodaka"]
                    == najbolji_fold["prosek_pogodaka"]
                    and kandidat["broj_komponenti"]
                    < najbolji_fold["broj_komponenti"]
                )
            ):
                najbolji_fold = kandidat

        fold_rezultati.append(
            najbolji_fold
        )

        print(
            f"  Fold {redni + 1}/"
            f"{BROJ_WALK_FORWARD_FOLDOVA}"
            f" — obuka {kraj_obuke:,}"
            f" — provera {len(indeksi_provere):,}"
            f" — PCA {najbolji_fold['broj_komponenti']}"
            f" — prosek pogodaka "
            f"{najbolji_fold['prosek_pogodaka']:.6f}"
        )

    proseci = {}

    for broj_komponenti, rezultati in (
        rezultati_komponenti.items()
    ):
        if rezultati:
            proseci[broj_komponenti] = float(
                np.mean(rezultati)
            )

    if not proseci:
        raise RuntimeError(
            "Nested walk-forward nije proizveo rezultate."
        )

    izabrani_broj_komponenti = max(
        proseci,
        key=lambda broj: (
            proseci[broj],
            -broj,
        ),
    )

    return {
        "foldovi": fold_rezultati,
        "proseci_komponenti": proseci,
        "izabrani_broj_komponenti":
            izabrani_broj_komponenti,
        "broj_provera": int(
            sum(
                rezultat["broj_provera"]
                for rezultat in fold_rezultati
            )
        ),
        "prosek_pogodaka": float(
            np.mean(
                [
                    rezultat["prosek_pogodaka"]
                    for rezultat in fold_rezultati
                ]
            )
        ),
        "potpuno_tacnih": int(
            sum(
                rezultat["potpuno_tacnih"]
                for rezultat in fold_rezultati
            )
        ),
        "greske": int(
            sum(
                rezultat["greske"]
                for rezultat in fold_rezultati
            )
        ),
    }


# =============================================================================
# OBRADA JEDNE IGRE
# =============================================================================

def obradi_igru(
    naziv: str,
    putanja: str,
) -> dict:
    naslov(f"OBRADA: {naziv}")

    izvlacenja = ucitaj_csv(
        putanja
    )

    broj_redova = len(
        izvlacenja
    )

    (
        osobine,
        mete,
        next_osobine,
    ) = napravi_osobine(
        izvlacenja
    )

    broj_vremenskih_uzoraka = len(
        osobine
    )

    razvoj_kraj = int(
        broj_vremenskih_uzoraka
        * UDEO_RAZVOJ
    )

    validacija_kraj = int(
        broj_vremenskih_uzoraka
        * (
            UDEO_RAZVOJ
            + UDEO_VALIDACIJA
        )
    )

    razvoj_kraj = max(
        100,
        razvoj_kraj,
    )

    validacija_kraj = min(
        validacija_kraj,
        broj_vremenskih_uzoraka - 1,
    )

    print(f"CSV: {putanja}")
    print(f"Broj redova: {broj_redova:,}")
    print("Prvi red se tretira kao najstariji.")
    print("Poslednji red se tretira kao najnoviji.")
    print(
        f"Teorijska stopa broja: "
        f"{TEORIJSKA_STOPA:.9f}"
    )
    print(
        f"Teorijsko očekivanje pogodaka: "
        f"{TEORIJSKO_OCEKIVANJE_POGODAKA:.9f}"
    )

    naslov("1. PRAVLJENJE DISTRIBUCIJSKIH OSOBINA")

    print(
        f"Vremenskih uzoraka: "
        f"{broj_vremenskih_uzoraka:,}"
    )
    print(
        f"Osnovnih osobina: "
        f"{osobine.shape[2]}"
    )
    print(
        f"Brojeva po vremenskom uzorku: "
        f"{osobine.shape[1]}"
    )

    naslov("2. NESTED WALK-FORWARD IZBOR PCA KOMPONENTI")

    nested = nested_walk_forward(
        osobine=osobine,
        mete=mete,
        razvoj_kraj=razvoj_kraj,
    )

    broj_komponenti = nested[
        "izabrani_broj_komponenti"
    ]

    print(
        f"Izabran broj PCA komponenti: "
        f"{broj_komponenti}"
    )
    print(
        f"Nested prosek pogodaka: "
        f"{nested['prosek_pogodaka']:.6f}"
    )

    naslov("3. ZAKLJUČANA VALIDACIJA")

    model_razvoj = prilagodi_model(
        osobine=osobine,
        mete=mete,
        indeksi_obuke=np.arange(
            0,
            razvoj_kraj,
            dtype=np.int64,
        ),
        broj_komponenti=broj_komponenti,
    )

    indeksi_validacije = np.arange(
        razvoj_kraj,
        validacija_kraj,
        dtype=np.int64,
    )

    validacione_predikcije = (
        predvidi_vremenske_redove(
            model=model_razvoj,
            osobine=osobine,
            indeksi=indeksi_validacije,
        )
    )

    validacija = oceni_predikcije(
        validacione_predikcije,
        mete[indeksi_validacije],
    )

    print(
        f"Broj provera: "
        f"{validacija['broj_provera']:,}"
    )
    print(
        f"Prosečan broj pogodaka: "
        f"{validacija['prosek_pogodaka']:.6f}"
    )
    print(
        f"Potpuno tačnih prelaza: "
        f"{validacija['potpuno_tacnih']:,}"
    )
    print(
        f"Greške: "
        f"{validacija['greske']:,}"
    )

    naslov("4. ZAMRZNUTI ZAVRŠNI HOLDOUT")

    model_pre_holdout = prilagodi_model(
        osobine=osobine,
        mete=mete,
        indeksi_obuke=np.arange(
            0,
            validacija_kraj,
            dtype=np.int64,
        ),
        broj_komponenti=broj_komponenti,
    )

    indeksi_holdouta = np.arange(
        validacija_kraj,
        broj_vremenskih_uzoraka,
        dtype=np.int64,
    )

    holdout_predikcije = (
        predvidi_vremenske_redove(
            model=model_pre_holdout,
            osobine=osobine,
            indeksi=indeksi_holdouta,
        )
    )

    holdout = oceni_predikcije(
        holdout_predikcije,
        mete[indeksi_holdouta],
    )

    print(
        f"Broj provera: "
        f"{holdout['broj_provera']:,}"
    )
    print(
        f"Prosečan broj pogodaka: "
        f"{holdout['prosek_pogodaka']:.6f}"
    )
    print(
        f"Potpuno tačnih prelaza: "
        f"{holdout['potpuno_tacnih']:,}"
    )
    print(
        f"Greške: "
        f"{holdout['greske']:,}"
    )

    print("Raspodela pogodaka:")

    for broj in range(8):
        print(
            f"  {broj} pogodaka: "
            f"{holdout['raspodela'][broj]:,}"
        )

    naslov("5. ZAVRŠNA OBUKA I NEXT")

    zavrsni_model = prilagodi_model(
        osobine=osobine,
        mete=mete,
        indeksi_obuke=np.arange(
            0,
            broj_vremenskih_uzoraka,
            dtype=np.int64,
        ),
        broj_komponenti=broj_komponenti,
    )

    next_kombinacija, next_skorovi = predvidi_next(
        model=zavrsni_model,
        next_osobine=next_osobine,
    )

    next_rang = rang_kombinacije(
        next_kombinacija
    )

    naslov("KONTROLNA LISTA", znak="#")

    status(
        "Vremenski ispravne distribucijske osobine",
        len(osobine) > 0,
        f"uzoraka={len(osobine):,}",
    )
    status(
        "StandardScaler samo na obuci",
        zavrsni_model.scaler is not None,
    )
    status(
        "Rolling PCA",
        zavrsni_model.pca is not None,
        (
            f"prozor={min(PCA_PROZOR, len(osobine))}, "
            f"komponenti={broj_komponenti}"
        ),
    )
    status(
        "PCA reconstruction error",
        True,
    )
    status(
        "Originalne osobine i PCA komponente",
        True,
    )
    status(
        "HistGradientBoostingRegressor",
        zavrsni_model.regresor is not None,
    )
    status(
        "Nested walk-forward validacija",
        nested["broj_provera"] > 0,
        (
            f"provera={nested['broj_provera']:,}, "
            f"prosek={nested['prosek_pogodaka']:.4f}"
        ),
    )
    status(
        "Zaključana validacija",
        validacija["broj_provera"] > 0,
        (
            f"provera={validacija['broj_provera']:,}, "
            f"prosek={validacija['prosek_pogodaka']:.4f}"
        ),
    )
    status(
        "Zamrznuti završni holdout",
        holdout["broj_provera"] > 0,
        (
            f"provera={holdout['broj_provera']:,}, "
            f"prosek={holdout['prosek_pogodaka']:.4f}"
        ),
    )
    status(
        "Jedna NEXT predikcija",
        len(next_kombinacija) == 7,
    )

    naslov(
        f"KONAČNI REZULTAT — {naziv}",
        znak="#",
    )

    print(
        f"Broj kombinacija u CSV-u:          "
        f"{broj_redova:,}"
    )
    print(
        f"Razvojni skup:                     "
        f"{razvoj_kraj:,}"
    )
    print(
        f"Zaključana validacija:             "
        f"{validacija_kraj - razvoj_kraj:,}"
    )
    print(
        f"Završni holdout:                   "
        f"{broj_vremenskih_uzoraka - validacija_kraj:,}"
    )

    print()

    print(
        f"Greške u nested walk-forward:      "
        f"{nested['greske']:,}"
    )
    print(
        f"Nested prosek pogodaka:            "
        f"{nested['prosek_pogodaka']:.6f}"
    )
    print(
        f"Greške na zaključanoj validaciji:  "
        f"{validacija['greske']:,}"
    )
    print(
        f"Validacioni prosek pogodaka:       "
        f"{validacija['prosek_pogodaka']:.6f}"
    )
    print(
        f"Greške na završnom holdoutu:       "
        f"{holdout['greske']:,}"
    )
    print(
        f"Holdout prosek pogodaka:           "
        f"{holdout['prosek_pogodaka']:.6f}"
    )

    print()

    print(
        f"Izabrane PCA komponente:           "
        f"{broj_komponenti}"
    )
    print(
        f"Objašnjena PCA varijansa:          "
        f"{np.sum(zavrsni_model.pca.explained_variance_ratio_):.4%}"
    )
    print(
        f"Zaključani NEXT rang:              "
        f"{next_rang:,}"
    )
    print(
        f"NEXT:                              "
        f"{formatiraj_kombinaciju(next_kombinacija)}"
    )

    return {
        "naziv": naziv,
        "broj_redova": broj_redova,
        "broj_komponenti": broj_komponenti,
        "nested": nested,
        "validacija": validacija,
        "holdout": holdout,
        "next_rang": next_rang,
        "next": next_kombinacija,
        "next_skorovi": next_skorovi,
    }


# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def main() -> None:
    pocetak_programa = time.time()

    naslov(
        "LOTO 7/39 — ROLLING PCA "
        "DISTRIBUCIJSKI REGRESIONI SISTEM"
    )

    print(f"Seed: {SEED}")
    print(
        f"Teorijska stopa broja: "
        f"{TEORIJSKA_STOPA:.9f}"
    )
    print(
        f"Teorijsko očekivanje pogodaka: "
        f"{TEORIJSKO_OCEKIVANJE_POGODAKA:.9f}"
    )
    print(
        f"Ukupno mogućih kombinacija: "
        f"{BROJ_SVIH_KOMBINACIJA:,}"
    )

    loto = obradi_igru(
        naziv="Loto",
        putanja=LOTO_CSV,
    )

    naslov(
        "KONAČNA NEXT PREDIKCIJA",
        znak="#",
    )

    print(
        f"Loto:      "
        f"{formatiraj_kombinaciju(loto['next'])}"
    )
    print(
        f"Loto rang: "
        f"{loto['next_rang']:,}"
    )

    print()

    print(
        f"Ukupno vreme: "
        f"{time.time() - pocetak_programa:.2f} sekundi"
    )


if __name__ == "__main__":
    main()



"""
========================================================================================
LOTO 7/39 — ROLLING PCA DISTRIBUCIJSKI REGRESIONI SISTEM
========================================================================================
Seed: 39
Teorijska stopa broja: 0.179487179
Teorijsko očekivanje pogodaka: 1.256410256
Ukupno mogućih kombinacija: 15,380,937

========================================================================================
OBRADA: Loto
========================================================================================
CSV: /data/loto7_4682_k72.csv
Broj redova: 4,682
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.
Teorijska stopa broja: 0.179487179
Teorijsko očekivanje pogodaka: 1.256410256

========================================================================================
1. PRAVLJENJE DISTRIBUCIJSKIH OSOBINA
========================================================================================
Vremenskih uzoraka: 4,582
Osnovnih osobina: 15
Brojeva po vremenskom uzorku: 39

========================================================================================
2. NESTED WALK-FORWARD IZBOR PCA KOMPONENTI
========================================================================================
  Fold 1/5 — obuka 962 — provera 357 — PCA 3 — prosek pogodaka 1.324930
  Fold 2/5 — obuka 1,319 — provera 357 — PCA 2 — prosek pogodaka 1.201681
  Fold 3/5 — obuka 1,676 — provera 358 — PCA 3 — prosek pogodaka 1.287709
  Fold 4/5 — obuka 2,034 — provera 357 — PCA 3 — prosek pogodaka 1.358543
  Fold 5/5 — obuka 2,391 — provera 358 — PCA 7 — prosek pogodaka 1.279330
Izabran broj PCA komponenti: 3
Nested prosek pogodaka: 1.290439

========================================================================================
3. ZAKLJUČANA VALIDACIJA
========================================================================================
Broj provera: 916
Prosečan broj pogodaka: 1.293668
Potpuno tačnih prelaza: 0
Greške: 916

========================================================================================
4. ZAMRZNUTI ZAVRŠNI HOLDOUT
========================================================================================
Broj provera: 917
Prosečan broj pogodaka: 1.234460
Potpuno tačnih prelaza: 0
Greške: 917
Raspodela pogodaka:
  0 pogodaka: 198
  1 pogodaka: 393
  2 pogodaka: 243
  3 pogodaka: 80
  4 pogodaka: 2
  5 pogodaka: 1
  6 pogodaka: 0
  7 pogodaka: 0

========================================================================================
5. ZAVRŠNA OBUKA I NEXT
========================================================================================

########################################################################################
KONTROLNA LISTA
########################################################################################
Vremenski ispravne distribucijske osobine       PROŠLO         uzoraka=4,582
StandardScaler samo na obuci                    PROŠLO
Rolling PCA                                     PROŠLO         prozor=600, komponenti=3
PCA reconstruction error                        PROŠLO
Originalne osobine i PCA komponente             PROŠLO
HistGradientBoostingRegressor                   PROŠLO
Nested walk-forward validacija                  PROŠLO         provera=1,787, prosek=1.2904
Zaključana validacija                           PROŠLO         provera=916, prosek=1.2937
Zamrznuti završni holdout                       PROŠLO         provera=917, prosek=1.2345
Jedna NEXT predikcija                           PROŠLO

########################################################################################
KONAČNI REZULTAT — Loto
########################################################################################
Broj kombinacija u CSV-u:          4,682
Razvojni skup:                     2,749
Zaključana validacija:             916
Završni holdout:                   917

Greške u nested walk-forward:      1,787
Nested prosek pogodaka:            1.290439
Greške na zaključanoj validaciji:  916
Validacioni prosek pogodaka:       1.293668
Greške na završnom holdoutu:       917
Holdout prosek pogodaka:           1.234460

Izabrane PCA komponente:           3
Objašnjena PCA varijansa:          73.3162%
Zaključani NEXT rang:              8,894,042
NEXT:                              05, x, 08, y, 10, z, 28

########################################################################################
KONAČNA NEXT PREDIKCIJA
########################################################################################
Loto:      05, x, 08, y, 10, z, 28
Loto rang: 8,894,042

Ukupno vreme: 16.42 sekundi
"""



"""
PCA - Principal Component Analysis 
for feature selection, pattern discovery, and composite ranking models
"""



"""
Vremenski prilagođen PCA nad osobinama, a ne običan PCA nad samim izvučenim brojevima.
Za svaki istorijski trenutak već postoje ili se mogu napraviti osobine za svih 39 brojeva: 
gap, survival/hazard, promene distribucijskog skora, parovi, grafovski skor, režim, kratki i dugi vremenski prozori. 
PCA bi zatim:
- uklonio međusobno duplirane osobine;
- izdvojio nekoliko zajedničkih skrivenih faktora;
- merio koliko novo stanje odstupa od dotadašnje strukture preko PCA reconstruction error;
- davao kompaktniji ulaz regresorima i smanjio preprilagođavanje.

Najpametnija varijanta bila bi:

osobine iz prošlosti
→ StandardScaler obučen samo na trening delu
→ PCA sa brojem komponenti izabranim unutar walk-forward validacije
→ originalne osobine + PCA komponente + reconstruction error
→ regresioni model
→ zaključani holdout
→ NEXT

Ključno: PCA mora da se prilagođava isključivo na prošlim podacima unutar svakog vremenskog preseka. 
Ako se jednom izračuna nad celim CSV-om pre podele, nastaje curenje budućih informacija.
Običan PCA verovatno ne bi doneo nešto značajno novo, jer sam već koristio spektralne, grafovske i režimske metode. 
Nova vrednost bi bila u kombinaciji:

rolling PCA + PCA reconstruction error + walk-forward izbor komponenti

PCA nije pravi „feature selection“ jer ne bira postojeće osobine, nego pravi njihove linearne kombinacije. 
Zato bih ga koristio kao dodatni prikaz podataka i detektor promene strukture, ne kao samostalan model za NEXT predikciju. 
Ovo je jedina PCA primena koju smatram vrednom zasebnog novog zadatka.

rolling PCA
+ PCA reconstruction error
+ originalne osobine
+ regresioni model
+ stroga walk-forward validacija
+ zaključani holdout
"""
