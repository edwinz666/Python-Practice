from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from minimal_plugin._internal import __version__ as __version__
from minimal_plugin.utils import parse_into_expr, register_plugin, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

def pig_latinnify(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        symbol="pig_latinnify",
        is_elementwise=True,
        lib=lib,
    )

def noop(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="noop",
        is_elementwise=True,
    )

def abs_i64(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="abs_i64",
        is_elementwise=True,
    )

    
def abs_numeric(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="abs_numeric",
        is_elementwise=True,
    )
    
def sum_i64(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr, other],
        lib=lib,
        symbol="sum_i64",
        is_elementwise=True,
    )
    
def cum_sum(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="cum_sum",
        # note how this is False compared to the others
        is_elementwise=False,
    )
    
def cum_sum_elemwise(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="cum_sum_elemwise",
        # note how this is False compared to the others
        is_elementwise=True,
    )
    
    
def snowball_stem(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="snowball_stem",
        is_elementwise=True,
    )

def add_suffix(expr: IntoExpr, *, suffix: str) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="add_suffix",
        is_elementwise=True,
        kwargs={"suffix": suffix},
    )
    
def weighted_mean(expr: IntoExpr, weights: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr, weights],
        lib=lib,
        symbol="weighted_mean",
        is_elementwise=True,
    )
    
def non_zero_indices(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr], lib=lib, symbol="non_zero_indices", is_elementwise=True
    )
    
def shift_struct(expr: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[expr],
        lib=lib,
        symbol="shift_struct",
        is_elementwise=True,
    )
    
def reverse_geocode(lat: IntoExpr, long: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[lat, long], lib=lib, symbol="reverse_geocode", is_elementwise=True
    )
    
def vertical_weighted_mean(values: IntoExpr, weights: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[values, weights],
        lib=lib,
        symbol="vertical_weighted_mean",
        is_elementwise=False,
        returns_scalar=True,
    )