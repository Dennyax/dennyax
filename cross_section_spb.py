import os
import datetime as dt
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

main_dir = "/Users/dgranin/Research/Activity/WRF-Chem_out/"
data1 = 'wrfout_meteo_2021-07_urb_modif.nc'
data2 = 'wrfout_meteo_2021-07_no_urb.nc'
data3 = 'wrfout_meteo_2021-07_no_spb.nc'
save_dir = "/Users/dgranin/Research/Activity/WRF-Chem_out/Output/"

# Какую разность строим: ΔT = эксперимент - контроль
file_exp, label_exp = data1, 'urb_modif'
file_ctrl, label_ctrl = data3, 'no_spb'

# 'T' - стандартная переменная WRF (возмущение потенц. температуры, θ - 300 К):
# температура считается из T, P, PB. Любое другое имя (например 'TK') читается как есть.
var_name = 'T'

# Разрез по параллели через Санкт-Петербург
target_lat = 59.9343
target_lon = 30.3351
lon_min = 28.0
lon_max = 32.5

# Вертикаль: высота над поверхностью (из PHB); если PHB нет в файле - номера уровней модели
z_top = 3000.0      # верх разреза, м
n_levels = 30       # верх разреза, если PHB нет, номер уровня

# Сроки для осреднения за месяц
hours_sel = [2, 8, 14, 20]
utc_shift = 0       # 0 - сроки в UTC; 3 - местное время (UTC+3)

# Используются только если в файле нет переменной 'Times'
start_date = dt.datetime(2021, 7, 1, 0)
step_hours = 1

save_fig = True

G = 9.81
R_D = 287.0
CP = 1004.5


def read_times(ds):
    if 'Times' in ds.variables:
        s = nc.chartostring(ds.variables['Times'][:])
        return pd.DatetimeIndex(pd.to_datetime([x.strip() for x in s], format='%Y-%m-%d_%H:%M:%S'))
    n = len(ds.dimensions['Time'])
    return pd.DatetimeIndex(start_date + pd.to_timedelta(np.arange(n) * step_hours, unit='h'))


def static_field(ds, name, ndim):
    """Поле, не зависящее от времени: берём первый срок. ndim - размерность без времени."""
    var = ds.variables[name]
    a = var[0] if var.ndim == ndim + 1 else var[:]
    return np.ma.filled(a, np.nan)


def find_section(lat2d, lon2d, lat0, lon_a, lon_b):
    """Ячейки сетки вдоль параллели lat0 от lon_a до lon_b (ближайший сосед).
    В WRF сетка криволинейная, поэтому ищем по двумерным XLAT/XLONG."""
    coslat = np.cos(np.deg2rad(lat0))
    dlon = np.nanmedian(np.abs(np.diff(lon2d, axis=1)))
    dlat = np.nanmedian(np.abs(np.diff(lat2d, axis=0)))
    max_d2 = (1.5 * max(dlat, dlon * coslat)) ** 2   # дальше - точка вне области модели
    jj, ii = [], []
    for lon in np.arange(lon_a, lon_b + dlon / 4, dlon / 2):
        d2 = (lat2d - lat0) ** 2 + ((lon2d - lon) * coslat) ** 2
        j, i = np.unravel_index(np.argmin(d2), d2.shape)
        if d2[j, i] > max_d2:
            continue
        if not jj or (j, i) != (jj[-1], ii[-1]):
            jj.append(j)
            ii.append(i)
    if not jj:
        raise ValueError('Разрез целиком вне области модели')
    return np.array(jj), np.array(ii)


def read_temp(ds, t, nz, window):
    j0, j1, i0, i1 = window
    sl = (t, slice(0, nz), slice(j0, j1), slice(i0, i1))
    if var_name == 'T':
        theta = ds.variables['T'][sl] + 300.0
        p = ds.variables['P'][sl] + ds.variables['PB'][sl]
        return np.ma.filled(theta * (p / 1.0e5) ** (R_D / CP), np.nan)   # К
    return np.ma.filled(ds.variables[var_name][sl], np.nan)


def hourly_means(path, hour_idx, nz, window):
    """Среднее по месяцу отдельно для каждого срока: {час: массив (nz, ny_window, nx_window)}."""
    means = {}
    with nc.Dataset(path, 'r') as ds:
        for h, idx in hour_idx.items():
            acc = 0.0
            for t in idx:
                acc = acc + read_temp(ds, t, nz, window)
            means[h] = acc / len(idx)
    return means


with nc.Dataset(f'{main_dir}{file_ctrl}', 'r') as ds:
    lat2d = static_field(ds, 'XLAT', 2)
    lon2d = static_field(ds, 'XLONG', 2)
    times = read_times(ds)

    jj, ii = find_section(lat2d, lon2d, target_lat, lon_min, lon_max)
    lons = lon2d[jj, ii]
    order = np.argsort(lons)
    jj, ii, lons = jj[order], ii[order], lons[order]

    if 'PHB' in ds.variables:
        # высота уровней модели над поверхностью по базовому состоянию (одинакова в обоих экспериментах)
        phb = static_field(ds, 'PHB', 3)[:, jj, ii] / G
        z = 0.5 * (phb[:-1] + phb[1:]) - phb[0]
        y_label, y_top, y_bottom = 'Height above ground, m', z_top, 0
    else:
        nz_all = len(ds.dimensions['bottom_top'])
        z = np.repeat(np.arange(1, nz_all + 1)[:, None], len(jj), axis=1)
        y_label, y_top, y_bottom = 'Model levels', n_levels, 1

with nc.Dataset(f'{main_dir}{file_exp}', 'r') as ds:
    if not read_times(ds).equals(times):
        raise ValueError('Временные оси экспериментов не совпадают')

# Читаем только уровни ниже верха разреза и только прямоугольник вокруг разреза
above = np.where(z.min(axis=1) > y_top)[0]
nz_use = above[0] + 1 if above.size else z.shape[0]
z = z[:nz_use]
window = (jj.min(), jj.max() + 1, ii.min(), ii.max() + 1)

# Сроки: в файле время UTC, при необходимости сдвигаем на местное
local_times = times + pd.Timedelta(hours=utc_shift)
tz_label = 'UTC' if utc_shift == 0 else f'UTC{utc_shift:+d}'
hour_idx = {h: np.where((local_times.hour == h) & (local_times.minute == 0))[0] for h in hours_sel}
for h, idx in hour_idx.items():
    if idx.size == 0:
        raise ValueError(f'В данных нет срока {h:02d} {tz_label}')

print(f'Целевая широта: {target_lat}, '
      f'в модели вдоль разреза: {lat2d[jj, ii].min():.4f} - {lat2d[jj, ii].max():.4f}')
print(f'Выбранный диапазон долгот: {lons[0]:.4f} - {lons[-1]:.4f} ({len(jj)} точек)')
print(f'Верх разреза: {y_top:.0f} ({y_label}), уровней модели: {nz_use}')
print(f'Осреднение по периоду: {times[0].strftime("%d-%m-%Y %H:00")} - '
      f'{times[-1].strftime("%d-%m-%Y %H:00")} UTC')
print('Число сроков в среднем: ' + ', '.join(f'{h:02d} {tz_label} - {idx.size}' for h, idx in hour_idx.items()))

mean_exp = hourly_means(f'{main_dir}{file_exp}', hour_idx, nz_use, window)
mean_ctrl = hourly_means(f'{main_dir}{file_ctrl}', hour_idx, nz_use, window)

# Разность средних = среднее разностей; берём только ячейки разреза: (уровень, точка)
params = {h: (mean_exp[h] - mean_ctrl[h])[:, jj - window[0], ii - window[2]] for h in hours_sel}

Lons = np.tile(lons, (nz_use, 1))

# Общие симметричные уровни для всех сроков (одна шкала на 4 панели)
step = 0.2
vmax = max(np.ceil(max(np.nanmax(np.abs(p)) for p in params.values()) / step) * step, step)
contour_levels = np.arange(-vmax, vmax + step / 2, step)
# contour_levels = np.arange(-0.5, 1.1, 0.2)   # или вручную

fig, axes = plt.subplots(len(hours_sel), 1, figsize=(12, 3.8 * len(hours_sel)),
                         sharex=True, sharey=True, layout='constrained')

for ax, h in zip(axes, hours_sel):
    ax.set_axisbelow(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)

    cf = ax.contourf(Lons, z, params[h],
                     levels=contour_levels,
                     cmap='coolwarm',
                     extend='both',
                     alpha=0.9)

    ax.axvline(target_lon, color='k', lw=1.2, ls='--')
    ax.set_title(f'{h:02d} {tz_label}', loc='left', fontsize=16, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=13, labelpad=10)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(y_bottom, y_top)
    ax.tick_params(axis='both', which='major', labelsize=16)

axes[0].annotate('Saint Petersburg', xy=(target_lon, y_top), xytext=(6, -6),
                 textcoords='offset points', ha='left', va='top', fontsize=13)
axes[-1].set_xlabel('Longitude, °', fontsize=13, labelpad=10)

# fig.suptitle(
#     f"Cross section along {target_lat}°N: \n Mean ΔT ([{label_exp}] - [{label_ctrl}])",
#     fontsize=20, fontweight='bold'
# )

cbar = fig.colorbar(cf, ax=axes, orientation='vertical', pad=0.02, shrink=0.9)
cbar.set_label('ΔT, °C', labelpad=10, size=13)
cbar.ax.tick_params(labelsize=14)

if save_fig:
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{save_dir}cross_section_spb_{label_exp}_minus_{label_ctrl}.png"
    fig.savefig(fname, bbox_inches='tight', dpi=300)
