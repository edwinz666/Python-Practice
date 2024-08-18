#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;
use std::borrow::Cow;

//////////////////// TAKE 1 //////////////////////
// #[polars_expr(output_type=String)]
// fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
//     let s = &inputs[0];
//     let ca = s.str()?;
//     let out: StringChunked = ca.apply(|opt_v: Option<&str>| {
//         opt_v.map(|value: &str| {
//             // Not the recommended way to do it,
//             // see below for a better way!
//             if let Some(first_char) = value.chars().next() {
//                 Cow::Owned(format!("{}{}ay", &value[1..], first_char))
//             } else {
//                 Cow::Borrowed(value)
//             }
//         })
//     });
//     Ok(out.into_series())
// }

//////////////////// TAKE 2 //////////////////////
#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(|value: &str, output: &mut String| {
        if let Some(first_char) = value.chars().next() {
            write!(output, "{}{}ay", &value[1..], first_char).unwrap()
        }
    });
    Ok(out.into_series())
}

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

#[polars_expr(output_type_func=same_output_type)]
fn noop(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    Ok(s.clone())
} 

//////////////////// TAKE 1 ////////////////////
// #[polars_expr(output_type=Int64)]
// fn abs_i64(inputs: &[Series]) -> PolarsResult<Series> {
//     let s = &inputs[0];
//     let ca: &Int64Chunked = s.i64()?;
//     // NOTE: there's a faster way of implementing `abs_i64`, which we'll
//     // cover in section 7.
//     let out: Int64Chunked = ca.apply(|opt_v: Option<i64>| opt_v.map(|v: i64| v.abs()));
//     Ok(out.into_series())
// }

//////////////////// TAKE 2 ////////////////////
#[polars_expr(output_type=Int64)]
fn abs_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.i64()?;
    let out = ca.apply_values(|x| x.abs());
    Ok(out.into_series())
}

//////////////////////////////////
use pyo3_polars::export::polars_core::export::num::Signed;
fn impl_abs_numeric<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Signed,
{
    // NOTE: there's a faster way of implementing `abs`, which we'll
    // cover in section 7.
    ca.apply(|opt_v: Option<T::Native>| opt_v.map(|v: T::Native| v.abs()))
}

#[polars_expr(output_type_func=same_output_type)]
fn abs_numeric(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::Int32 => Ok(impl_abs_numeric(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(impl_abs_numeric(s.i64().unwrap()).into_series()),
        DataType::Float32 => Ok(impl_abs_numeric(s.f32().unwrap()).into_series()),
        DataType::Float64 => Ok(impl_abs_numeric(s.f64().unwrap()).into_series()),
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype {dtype} not \
            supported for abs_numeric, expected Int32, Int64, Float32, Float64."))
        }
    }
}


//////////////////////// TAKE 1 ///////////////////////
// use polars::prelude::arity::broadcast_binary_elementwise;
// #[polars_expr(output_type=Int64)]
// fn sum_i64(inputs: &[Series]) -> PolarsResult<Series> {
//     let left: &Int64Chunked = inputs[0].i64()?;
//     let right: &Int64Chunked = inputs[1].i64()?;
//     // Note: there's a faster way of summing two columns, see
//     // section 7.
//     let out: Int64Chunked = broadcast_binary_elementwise(
//         left,
//         right,
//         |left: Option<i64>, right: Option<i64>| match (left, right) {
//             (Some(left), Some(right)) => Some(left + right),
//             _ => None,
//         },
//     );
//     Ok(out.into_series())
// }

//////////////////////// TAKE 2 ///////////////////////
use polars::prelude::arity::broadcast_binary_elementwise_values;
#[polars_expr(output_type=Int64)]
fn sum_i64(inputs: &[Series]) -> PolarsResult<Series> {
    let left: &Int64Chunked = inputs[0].i64()?;
    let right: &Int64Chunked = inputs[1].i64()?;
    let out: Int64Chunked = broadcast_binary_elementwise_values(
        left,
        right,
        // |left: i64, right: i64| Some(left + right)
        |left: i64, right: i64| left + right
    );
    Ok(out.into_series())
}

/////////////////////////////////////////////////////
use pyo3_polars::export::polars_core::utils::CustomIterTools;
#[polars_expr(output_type_func=same_output_type)]
fn cum_sum(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca: &Int64Chunked = s.i64()?;
    let out: Int64Chunked = ca
        .iter()
        .scan(0_i64, |state: &mut i64, x: Option<i64>| {
            match x {
                Some(x) => {
                    *state += x;
                    Some(Some(*state))
                },
                None => Some(None),
            }
        })
        .collect_trusted();
    let out: Int64Chunked = out.with_name(ca.name());
    Ok(out.into_series())
}

#[polars_expr(output_type_func=same_output_type)]
fn cum_sum_elemwise(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca: &Int64Chunked = s.i64()?;
    let out: Int64Chunked = ca
        .iter()
        .scan(0_i64, |state: &mut i64, x: Option<i64>| {
            match x {
                Some(x) => {
                    *state += x;
                    Some(Some(*state))
                },
                None => Some(None),
            }
        })
        .collect_trusted();
    let out: Int64Chunked = out.with_name(ca.name());
    Ok(out.into_series())
}

//////////////////////////////////////////////
use rust_stemmers::{Algorithm, Stemmer};
#[polars_expr(output_type=String)]
fn snowball_stem(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let en_stemmer = Stemmer::create(Algorithm::English);
    let out: StringChunked = ca.apply_to_buffer(|value: &str, output: &mut String| {
        write!(output, "{}", en_stemmer.stem(value)).unwrap()
    });
    Ok(out.into_series())
}

//////////////////////////////////////////////
use serde::Deserialize;
#[derive(Deserialize)]
struct AddSuffixKwargs {
    suffix: String,
}

#[polars_expr(output_type=String)]
fn add_suffix(inputs: &[Series], kwargs: AddSuffixKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ca = s.str()?;
    let out = ca.apply_to_buffer(|value, output| {
        write!(output, "{}{}", value, kwargs.suffix).unwrap();
    });
    Ok(out.into_series())
}

///////////////////////////////////////////
use crate::utils::binary_amortized_elementwise;
#[polars_expr(output_type=Float64)]
fn weighted_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let values = inputs[0].list()?;
    let weights = &inputs[1].list()?;

    let out: Float64Chunked = binary_amortized_elementwise(
        values,
        weights,
        |values_inner: &Series, weights_inner: &Series| -> Option<f64> {
            let values_inner = values_inner.i64().unwrap();
            let weights_inner = weights_inner.f64().unwrap();
            if values_inner.len() == 0 {
                // Mirror Polars, and return None for empty mean.
                return None
            }
            let mut numerator: f64 = 0.;
            let mut denominator: f64 = 0.;
            values_inner
                .iter()
                .zip(weights_inner.iter())
                .for_each(|(v, w)| {
                    if let (Some(v), Some(w)) = (v, w) {
                        numerator += v as f64 * w;
                        denominator += w;
                    }
                });
            Some(numerator / denominator)
        },
    );
    Ok(out.into_series())
}

/////////////////////////////////////////////////////
fn list_idx_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(input_fields[0].name(), DataType::List(Box::new(IDX_DTYPE)));
    Ok(field.clone())
}

#[polars_expr(output_type_func=list_idx_dtype)]
fn non_zero_indices(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;

    let out: ListChunked = ca.apply_amortized(|s| {
        let s: &Series = s.as_ref();
        let ca: &Int64Chunked = s.i64().unwrap();
        let out: IdxCa = ca
            .iter()
            .enumerate()
            .filter(|(_idx, opt_val)| opt_val != &Some(0))
            .map(|(idx, _opt_val)| Some(idx as IdxSize))
            .collect_ca("");
        out.into_series()
    });
    Ok(out.into_series())
}

/////////////////////////////////////////////////////////////
fn shifted_struct(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.data_type() {
        DataType::Struct(fields) => {
            let mut field_0 = fields[0].clone();
            let name = field_0.name().clone();
            field_0.set_name(fields[fields.len() - 1].name().clone());
            let mut fields = fields[1..]
                .iter()
                .zip(fields[0..fields.len() - 1].iter())
                .map(|(fld, name)| Field::new(name.name(), fld.data_type().clone()))
                .collect::<Vec<_>>();
            fields.push(field_0);
            Ok(Field::new(&name, DataType::Struct(fields)))
        }
        _ => unreachable!(),
    }
}

// note how after the # it refers to shifted_struct
#[polars_expr(output_type_func=shifted_struct)]
fn shift_struct(inputs: &[Series]) -> PolarsResult<Series> {
    let struct_ = inputs[0].struct_()?;
    let fields = struct_.fields();
    if fields.is_empty() {
        return Ok(inputs[0].clone());
    }
    let mut field_0 = fields[0].clone();
    field_0.rename(fields[fields.len() - 1].name());
    let mut fields = fields[1..]
        .iter()
        .zip(fields[..fields.len() - 1].iter())
        .map(|(s, name)| {
            let mut s = s.clone();
            s.rename(name.name());
            s
        })
        .collect::<Vec<_>>();
    fields.push(field_0);
    StructChunked::new(struct_.name(), &fields).map(|ca| ca.into_series())
}

///////////////////////////////////////////
/// MutablePlString https://pola.rs/posts/polars-string-type/
use polars_arrow::array::MutablePlString;
use polars_core::utils::align_chunks_binary;
use reverse_geocoder::ReverseGeocoder;

#[polars_expr(output_type=String)]
fn reverse_geocode(inputs: &[Series]) -> PolarsResult<Series> {
    let lat = inputs[0].f64()?;
    let lon = inputs[1].f64()?;
    let geocoder = ReverseGeocoder::new();

    let (lhs, rhs) = align_chunks_binary(lat, lon);
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lat_arr, lon_arr)| {
            let mut mutarr = MutablePlString::with_capacity(lat_arr.len());

            for (lat_opt_val, lon_opt_val) in lat_arr.iter().zip(lon_arr.iter()) {
                match (lat_opt_val, lon_opt_val) {
                    (Some(lat_val), Some(lon_val)) => {
                        let res = &geocoder.search((*lat_val, *lon_val)).record.name;
                        mutarr.push(Some(res))
                    }
                    _ => mutarr.push_null(),
                }
            }

            mutarr.freeze().boxed()
        })
        .collect();
    let out: StringChunked = unsafe { ChunkedArray::from_chunks("placeholder", chunks) };
    Ok(out.into_series())
}

///////////////////////////////////////
#[polars_expr(output_type=Float64)]
fn vertical_weighted_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let values = &inputs[0].f64()?;
    let weights = &inputs[1].f64()?;
    let mut numerator = 0.;
    let mut denominator = 0.;
    values.iter().zip(weights.iter()).for_each(|(v, w)| {
        if let (Some(v), Some(w)) = (v, w) {
            numerator += v * w;
            denominator += w;
        }
    });
    let result = numerator / denominator;
    Ok(Series::new("", vec![result]))
}