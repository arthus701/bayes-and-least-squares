import numpy as np
import pandas as pd

from warnings import warn


currentyear = np.nan

tab = pd.read_csv(
    '../dat/arch10k.txt',
    sep=r'\s+',
    na_values=[9999],
)

rename_dict = {
    '%Age': 't',
    'dAge': 'dt',
    'SiteLat': 'lat',
    'SiteLon': 'lon',
    'Decl': 'D',
    'Incl': 'I',
    'Ba': 'F',
    'dBa': 'dF',
}
tab.rename(rename_dict, inplace=True, axis='columns')

tab['colat'] = 90 - tab['lat']
tab['rad'] = 6371.2
tab['dt'] = 100
tab['FID'] = "ERDA/2206/"

tab['dI'] = tab['alpha95']
# tab['dI'] *= 57.3/140.
# Use wrong formula on purpose
tab['dI'] *= 81/140.
tab['dI'] = tab['dI'].where(
    tab['I'].isna() | tab['dI'].notna(),
    other=1.9,
)

# tab['dI'] = tab['dI'].where(
#     tab['dI'] != 0,
#     other=1.9,
# )
tab['dI'] = tab['dI'].where(
    tab['dI'] > 1.9,
    other=1.9,
)

tab['dD'] = tab['dI']

# Find records of only Declination, since this causes trouble in the error
# calculation
cond = tab['D'].notna() & tab['I'].isna()
# Get the corresponding indices
ind = tab.where(cond).dropna(how='all').index
# If there are indices in the array, throw a warning.
if ind.size != 0:
    warn(
        f"Records with indices {ind.values} contain declination, but not"
        " inclination! The errors need special treatment!\n"
        "To be able to use the provided data, these"
        " records have been dropped from the output.",
        UserWarning
    )

tab.drop(tab.where(cond).dropna(how='all').index, inplace=True)

tab['dD'] /= np.cos(np.deg2rad(tab['I']))

tab['dF'] = tab['dF'].where(
    tab['F'].isna() | tab['dF'].notna(),
    other=5.,
)
# tab['dF'] = tab['dF'].where(
#     tab['dF'] != 0,
#     other=5.,
# )
tab['dF'] = tab['dF'].where(
    tab['dF'] > 5.,
    other=5.,
)

tab['dF'] /= 2
dt = np.copy(tab.loc[:, 'dt'].values)

# unplausible lat record removed
# tab.drop(index=4496, inplace=True)
tab.drop(columns=['alpha95', 'RefID', 'SiteID'], inplace=True)


tab.loc[:, 'dt'] = 0

if __name__ == '__main__':
    print(np.any(tab.dropna(subset=['D'])['dD'] == 0))
    print(np.any(np.isnan(tab.dropna(subset=['D'])['dD'])))
    print(np.any(tab.dropna(subset=['I'])['dI'] == 0))
    print(np.any(np.isnan(tab.dropna(subset=['I'])['dI'])))
    print(np.any(tab.dropna(subset=['F'])['dF'] == 0))
    print(np.any(np.isnan(tab.dropna(subset=['F'])['dF'])))
    print(tab.columns)
    print(tab)
