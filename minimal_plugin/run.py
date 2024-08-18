import polars as pl
from minimal_plugin import pig_latinnify
import minimal_plugin as mp

# df = pl.DataFrame({
#     'english': ['this', 'is', 'not', 'pig', 'latin'],
# })
# result = df.with_columns(pig_latin = pig_latinnify('english'))
# print(result)


# df = pl.DataFrame({
#     'a': [1, 1, None],
#     'b': [4.1, 5.2, 6.3],
#     'c': ['hello', 'everybody!', '!']
# })
# print(df.with_columns(mp.noop(pl.all()).name.suffix('_noop')))


# df = pl.DataFrame({
#     'a': [1, -1, None],
#     'b': [4.1, 5.2, -6.3],
#     'c': ['hello', 'everybody!', '!']
# })
# print(df.with_columns(mp.abs_i64('a').name.suffix('_abs')))
# print(df.with_columns(mp.abs_numeric(pl.col('a', 'b')).name.suffix('_abs')))


# df = pl.DataFrame({'a': [1, 5, 2], 'b': [3, None, -1]})
# print(df.with_columns(a_plus_b=mp.sum_i64('a', 'b')))
# print(df.with_columns(a_cum_sum=mp.cum_sum('a')))


# df = pl.DataFrame({
#     'a': [1, 2, 3, 4, None, 5],
#     'b': [1, 1, 1, 2, 2, 2],
# })
# print(df.with_columns(a_cum_sum=mp.cum_sum('a')))
# print(df.with_columns(a_cum_sum=mp.cum_sum_elemwise('a')))
# print(df.with_columns(a_cum_sum=mp.cum_sum('a').over('b')))
# print(df.with_columns(a_cum_sum=mp.cum_sum_elemwise('a').over('b')))

# df = pl.DataFrame({'a': ["I", "love", "pig", "latin"]})
# print(df.with_columns(a_pig_latin=mp.pig_latinnify('a')))


# df = pl.DataFrame({'word': ["fearlessly", "littleness", "lovingly", "devoted"]})
# print(df.with_columns(b=mp.snowball_stem('word')))


# df = pl.DataFrame({'a': ['bob', 'billy']})
# print(df.with_columns(mp.add_suffix('a', suffix='-billy')))


# df = pl.DataFrame({
#     'values': [[1, 3, 2], [5, 7]],
#     'weights': [[.5, .3, .2], [.1, .9]]
# })
# print(df.with_columns(weighted_mean = mp.weighted_mean('values', 'weights')))


# pl.Config().set_fmt_table_cell_list_len(10)
# df = pl.DataFrame({'dense': [[0, 9], [8, 6, 0, 9], None, [3, 3]]})
# print(df)
# print(df.with_columns(indices=mp.non_zero_indices('dense')))


# df = pl.DataFrame(
#     {
#         "a": [1, 3, 8],
#         "b": [2.0, 3.1, 2.5],
#         "c": ["3", "7", "3"],
#     }
# ).select(abc=pl.struct("a", "b", "c"))
# print(df.with_columns(abc_shifted=mp.shift_struct("abc")))


# df = pl.DataFrame({
#     'lat': [37.7749, 51.01, 52.5],
#     'lon': [-122.4194, -3.9, -.91]
# })
# print(df.with_columns(city=mp.reverse_geocode('lat', 'lon')))


df = pl.DataFrame({
    'values': [1., 3, 2, 5, 7],
    'weights': [.5, .3, .2, .1, .9],
    'group': ['a', 'a', 'a', 'b', 'b'],
})
print(df.group_by('group').agg(weighted_mean = mp.vertical_weighted_mean('values', 'weights')))