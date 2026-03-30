"""
core/global_config.py
=====================
Singleton accessor for data/global_config.json — the single source of truth
for benchmark-point values, observational data, SIDM cuts, shared CSV paths,
and physical/cosmological constants.

Usage (from any script that already has core/ on sys.path):

    from global_config import GC

    bp1  = GC.benchmark("BP1")           # {"label":"BP1","m_chi_GeV":...}
    bps  = GC.benchmarks("BP1","MAP")    # list of dicts
    bps  = GC.all_benchmarks()           # all five named BPs

    obs  = GC.observations()             # list of 13 dicts
    obs  = GC.observations_as_tuples()   # list of (name,v,central,lo,hi,ref)

    cuts = GC.sidm_cuts()                # {"sigma_m_30_lo":0.5, ...}
    path = GC.csv_path("relic_viable")   # pathlib.Path (absolute)
    c    = GC.cosmological_constants()   # cosmological constants dict
    pc   = GC.physical_constants()       # universal physical constants dict
    fh   = GC.fornax_halo()             # Fornax halo parameters
    w09  = GC.walker2009_fornax()       # Walker+2009 binned σ_los data

Benchmark labels available: BP1, BP9, BP16, MAP, MAP_relic
"""
import json
from pathlib import Path

# observational_database.json is at <repo_root>/data/observational_database.json
# This file lives at <repo_root>/chapter1_sidm/core/global_config.py
# So parent.parent.parent = the_dark_unification/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GLOBAL_CONFIG_PATH = _REPO_ROOT / "data" / "observational_database.json"


class _GlobalConfig:
    """Lazy-loading singleton for data/global_config.json."""

    def __init__(self):
        self._data = None

    def _load(self):
        if self._data is None:
            with open(_GLOBAL_CONFIG_PATH, "r", encoding="utf-8") as fh:
                self._data = json.load(fh)

    def _get(self, key):
        self._load()
        return self._data[key]

    # ------------------------------------------------------------------ #
    #  Benchmark points                                                    #
    # ------------------------------------------------------------------ #

    def benchmark(self, label: str) -> dict:
        """Return parameter dict for one named benchmark, with 'label' key added.

        Available: BP1, BP9, BP16, MAP, MAP_relic
        """
        bps = self._get("benchmark_points")
        if label not in bps:
            available = list(bps.keys())
            raise KeyError(
                f"Unknown benchmark label '{label}'. "
                f"Available: {available}"
            )
        bp = dict(bps[label])
        bp["label"] = label
        return bp

    def benchmarks(self, *labels) -> list:
        """Return a list of benchmark dicts for the requested labels."""
        return [self.benchmark(lbl) for lbl in labels]

    def all_benchmarks(self) -> list:
        """Return all five benchmark dicts (BP1, BP9, BP16, MAP, MAP_relic)."""
        bps = self._get("benchmark_points")
        return [dict(v, label=k) for k, v in bps.items()]

    def benchmarks_from_labels(self, labels: list) -> list:
        """Convenience: resolve a list of label strings to benchmark dicts."""
        return [self.benchmark(lbl) for lbl in labels]

    # ------------------------------------------------------------------ #
    #  Observational data                                                  #
    # ------------------------------------------------------------------ #

    def observations(self) -> list:
        """Return the 13-entry observational dataset as a list of dicts.

        Each dict has keys: name, v_km_s, central, lo, hi, ref
        """
        return list(self._get("observations"))

    def observations_as_tuples(self) -> list:
        """Return observations as list of (name, v_km_s, central, lo, hi, ref).

        Drop-in replacement for the legacy _DEFAULT_OBS hardcoded lists.
        """
        return [
            (o["name"], o["v_km_s"], o["central"], o["lo"], o["hi"], o["ref"])
            for o in self._get("observations")
        ]

    # ------------------------------------------------------------------ #
    #  SIDM cuts & configuration                                           #
    # ------------------------------------------------------------------ #

    def sidm_cuts(self) -> dict:
        """Return SIDM viability cuts: sigma_m_30_lo/hi, sigma_m_1000_hi."""
        return dict(self._get("sidm_cuts"))

    # ------------------------------------------------------------------ #
    #  Shared CSV paths                                                    #
    # ------------------------------------------------------------------ #

    def csv_path(self, name: str) -> Path:
        """Return absolute Path to a named shared CSV file in data/.

        Available keys: all_viable_repr, relic_viable, results, benchmarks_v30
        Maps to the canonical filenames copied to the_dark_unification/data/.
        """
        _map = {
            "all_viable_raw":   "all_viable_representative.csv",
            "all_viable_repr":  "all_viable_representative.csv",
            "relic_all":        "sidm_relic_viable_points.csv",
            "relic_viable":     "sidm_relic_viable_points.csv",
            "results":          "chi2_fit_results.csv",
            "benchmarks_v30":   "benchmark_points.csv",
        }
        if name not in _map:
            raise KeyError(f"Unknown CSV key '{name}'. Available: {list(_map)}")
        return _REPO_ROOT / "data" / _map[name]

    # ------------------------------------------------------------------ #
    #  Physical & cosmological constants                                   #
    # ------------------------------------------------------------------ #

    def physical_constants(self) -> dict:
        """Return universal physical constants dict (GEV2_to_cm2, GeV_in_g, etc.)."""
        return dict(self._get("physical_constants"))

    def cosmological_constants(self) -> dict:
        """Return cosmological constants dict (Planck 2018 + SM values)."""
        return dict(self._get("cosmological_constants"))

    def constants(self) -> dict:
        """Return cosmological constants dict. Alias for cosmological_constants()."""
        return self.cosmological_constants()

    # ------------------------------------------------------------------ #
    #  Astrophysical datasets                                              #
    # ------------------------------------------------------------------ #

    def fornax_halo(self) -> dict:
        """Return canonical Fornax dSph halo parameters (Read+2019, Walker+2009)."""
        return dict(self._get("fornax_halo"))

    def walker2009_fornax(self) -> dict:
        """Return Walker+2009 binned line-of-sight σ_los data for Fornax."""
        return dict(self._get("walker2009_fornax"))


# Module-level singleton — import and use directly
GC = _GlobalConfig()
