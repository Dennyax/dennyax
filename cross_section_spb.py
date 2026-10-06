import os
import json
import datetime as dt
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

repo_dir = os.path.dirname(os.path.abspath(__file__))

# Пути можно задать переменными окружения WRF_OUT_DIR и WRF_SAVE_DIR (удобно при запуске из клона репозитория)
main_dir = os.environ.get('WRF_OUT_DIR', "/Users/dgranin/Research/Activity/WRF-Chem_out/")
save_dir = os.environ.get('WRF_SAVE_DIR', os.path.join(repo_dir, 'output'))
data_dir = os.path.join(repo_dir, 'data')   # осреднённые разрезы (.npz) - маленькие, их можно коммитить
data1 = 'wrfout_meteo_2021-07_urb_modif.nc'
data2 = 'wrfout_meteo_2021-07_no_urb.nc'
data3 = 'wrfout_meteo_2021-07_no_spb.nc'

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

# True: если в data/ есть .npz с теми же параметрами, wrfout не читаем (рисунок строится из .npz)
use_cache = True
save_fig = True

G = 9.81
R_D = 287.0
CP = 1004.5

tz_label = 'UTC' if utc_shift == 0 else f'UTC{utc_shift:+d}'

# Параметры, от которых зависит содержимое .npz: при их изменении кэш пересчитывается
config = dict(file_exp=file_exp, file_ctrl=file_ctrl, var_name=var_name,
              target_lat=target_lat, lon_min=lon_min, lon_max=lon_max,
              z_top=z_top, n_levels=n_levels, hours_sel=hours_sel, utc_shift=utc_shift)
config_str = json.dumps(config, sort_keys=True)
npz_path = os.path.join(data_dir, f'section_spb_{label_exp}_minus_{label_ctrl}.npz')


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


def compute_section():
    """Читает wrfout и считает ΔT вдоль разреза для каждого срока."""
    path_ctrl = os.path.join(main_dir, file_ctrl)
    path_exp = os.path.join(main_dir, file_exp)

    with nc.Dataset(path_ctrl, 'r') as ds:
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

    with nc.Dataset(path_exp, 'r') as ds:
        if not read_times(ds).equals(times):
            raise ValueError('Временные оси экспериментов не совпадают')

    # Читаем только уровни ниже верха разреза и только прямоугольник вокруг разреза
    above = np.where(z.min(axis=1) > y_top)[0]
    nz_use = above[0] + 1 if above.size else z.shape[0]
    z = z[:nz_use]
    window = (jj.min(), jj.max() + 1, ii.min(), ii.max() + 1)

    # Сроки: в файле время UTC, при необходимости сдвигаем на местное
    local_times = times + pd.Timedelta(hours=utc_shift)
    hour_idx = {h: np.where((local_times.hour == h) & (local_times.minute == 0))[0] for h in hours_sel}
    for h, idx in hour_idx.items():
        if idx.size == 0:
            raise ValueError(f'В данных нет срока {h:02d} {tz_label}')

    period = (f'{times[0].strftime("%d-%m-%Y %H:00")} - {times[-1].strftime("%d-%m-%Y %H:00")} UTC')
    print(f'Целевая широта: {target_lat}, '
          f'в модели вдоль разреза: {lat2d[jj, ii].min():.4f} - {lat2d[jj, ii].max():.4f}')
    print(f'Выбранный диапазон долгот: {lons[0]:.4f} - {lons[-1]:.4f} ({len(jj)} точек)')
    print(f'Верх разреза: {y_top:.0f} ({y_label}), уровней модели: {nz_use}')
    print(f'Осреднение по периоду: {period}')
    print('Число сроков в среднем: '
          + ', '.join(f'{h:02d} {tz_label} - {idx.size}' for h, idx in hour_idx.items()))

    mean_exp = hourly_means(path_exp, hour_idx, nz_use, window)
    mean_ctrl = hourly_means(path_ctrl, hour_idx, nz_use, window)

    # Разность средних = среднее разностей; берём только ячейки разреза: (уровень, точка)
    params = np.stack([(mean_exp[h] - mean_ctrl[h])[:, jj - window[0], ii - window[2]] for h in hours_sel])

    return dict(lons=lons, z=z, params=params,
                counts=np.array([hour_idx[h].size for h in hours_sel]),
                y_label=y_label, y_top=y_top, y_bottom=y_bottom, period=period)


def save_section(section):
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(npz_path, config=config_str, **section)


def load_section():
    if not (use_cache and os.path.exists(npz_path)):
        return None
    with np.load(npz_path) as f:
        if str(f['config']) != config_str:
            print(f'Параметры в {npz_path} отличаются от текущих - пересчитываем из wrfout')
            return None
        section = {k: f[k] for k in f.files if k != 'config'}
    for k in ('y_label', 'period'):
        section[k] = str(section[k])
    for k in ('y_top', 'y_bottom'):
        section[k] = float(section[k])
    return section


section = load_section()
if section is None:
    section = compute_section()
    save_section(section)
    print(f'Осреднённые разрезы сохранены: {npz_path}')
else:
    print(f'Осреднённые разрезы загружены из {npz_path} (период: {section["period"]}), wrfout не читались')

lons, z, params = section['lons'], section['z'], section['params']
y_label, y_top, y_bottom = section['y_label'], section['y_top'], section['y_bottom']

Lons = np.tile(lons, (z.shape[0], 1))

# Общие симметричные уровни для всех сроков (одна шкала на 4 панели)
step = 0.2
vmax = max(np.ceil(np.nanmax(np.abs(params)) / step) * step, step)
contour_levels = np.arange(-vmax, vmax + step / 2, step)
# contour_levels = np.arange(-0.5, 1.1, 0.2)   # или вручную

fig, axes = plt.subplots(len(hours_sel), 1, figsize=(12, 3.8 * len(hours_sel)),
                         sharex=True, sharey=True, layout='constrained')

for ax, h, param in zip(axes, hours_sel, params):
    ax.set_axisbelow(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)

    cf = ax.contourf(Lons, z, param,
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
    fname = os.path.join(save_dir, f"cross_section_spb_{label_exp}_minus_{label_ctrl}.png")
    fig.savefig(fname, bbox_inches='tight', dpi=300)
