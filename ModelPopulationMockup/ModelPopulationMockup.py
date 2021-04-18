import pandas as pd
import numpy as np
from random import random, choice

class ModelPopulationMockup():

    def __init__(self, population_size, responder_size, ntiles, ks_target, psi_target):
        self._population_size = population_size
        self._responder_size = responder_size
        self._ntiles = ntiles
        self._ks_target = ks_target
        self._psi_target = psi_target

    def set_initial_population(self):
        # Define initial population
        initial_population = [ int(self._population_size / self._ntiles) for _ in range(self._ntiles) ]
        self._df = pd.DataFrame(initial_population, columns=['Population'])

    def set_responder_population(self):
        responder_population = [ int(self._responder_size / self._ntiles) for _ in range(self._ntiles) ]
        self._df['Responder'] = responder_population

    def calculate_psi(self, inset):
        # Calculate PSI
        df = inset.copy()
        df['Baseline'] = [ 0.1 for _ in range(self._ntiles) ]
        df['Population %'] = df['Population'] / self._population_size
        df['psi'] = (df['Population %'] - df['Baseline']) * np.log(df['Population %'] / df['Baseline'])
        return df['psi'].sum()

    def shuffle_deciles(self, inset, chunk_size = 0.1):
        df = inset.copy()
        list_of_ntiles = list(range(self._ntiles))
        for i in list_of_ntiles:
            r = random()
            if(r < 0.5):
                c = choice(list_of_ntiles)
                chunk = int(df.iloc[c]['Population'] * chunk_size)
                if chunk > 0:
                    df.iloc[i, 0] += chunk
                    df.iloc[c, 0] -= chunk
        return df

    def find_psi_target(self):
        df = inset.copy()
        i = 0
        while calculate_psi(df) < psi_target:
            df = self.shuffle_deciles(df, 0.01 + i * 0.0001)
            i += 1
        return df

    def calculate_ks(self, inset):
        df = inset.copy()
        df['Non Responder'] = df['Population'] - df['Responder']
        df['Responder %'] = df['Responder'] / df['Responder'].sum()
        df['Non Responder %'] = df['Non Responder'] / df['Non Responder'].sum()
        df = df.drop(['Population', 'Responder', 'Non Responder'], axis=1).cumsum()
        df['KS'] = df['Responder %'] - df['Non Responder %']
        return df['KS'].max()

    def shuffle_responders(self, inset):
        df = inset.copy()
        list_of_ntiles = list(range(self._ntiles))
        for i in list_of_ntiles:
            r = random()
            if(r < (1 - ( i * (1 / max(list_of_ntiles)) ))):
                c = choice(list_of_ntiles)
                chunk = int(df.iloc[c]['Responder'] * 0.1)
                # Prevent responder column to be greater than population column
                if df.iloc[i, 1] + chunk > df.iloc[i, 0]: continue
                if chunk > 0:
                    df.iloc[i, 1] += chunk
                    df.iloc[c, 1] -= chunk
        return df

    def find_ks_target(self):
        df = self._df.copy()
        while self.calculate_ks(df) < self._ks_target:
            df = self.shuffle_responders(df)
        self._df = df

    def gains_chart(self, inset, metrics = ['ks', 'psi', 'cumsum', 'odds', 'logodds']):

        df = inset.copy()
        if 'ks' in metrics:
            df['Non Responder'] = df['Population'] - df['Responder']
            df['Responder %'] = df['Responder'] / df['Responder'].sum()
            df['Non Responder %'] = df['Non Responder'] / df['Non Responder'].sum()
            ks = df[['Responder %', 'Non Responder %']]
            ks.rename({'Responder %': 'Cummulative Responder %', 'Non Responder %': 'Cummulative Non Responder %'})
            ks = ks.cumsum()
            ks['KS'] = ks['Cummulative Responder %'] - ks['Cummulative Non Responder %']
            df = pd.concat([df, ks])

        return df
            


    def run(self, verbose = False):
        self.set_initial_population()
        self.find_psi_target()
        self.set_responder_population()
        self.find_ks_target()

    def print_df(self):
        print(self._df)

    @property
    def df(self):
        return self._df

    @property
    def population_size(self):
        return self._population_size

    @property
    def responder_size(self):
        return self._responder_size

    @property
    def ntiles(self):
        return self._ntiles

    @property
    def ks_target(self):
        return self._ks_target

    @property
    def psi_target(self):
        return self._psi_target

    @classmethod
    def gains_chart(cls, inset, metrics = ['ks', 'psi', 'cumsum','odds', 'logodds', 'lift', 'separation']):
        df = inset.copy()
        
        if not 'Population' in df.columns or not 'Responder' in df.columns: return df
        
        if 'ks' in metrics:
            df['Non Responder'] = df['Population'] - df['Responder']
            df['Responder %'] = df['Responder'] / df['Responder'].sum()
            df['Non Responder %'] = df['Non Responder'] / df['Non Responder'].sum()
            ks = df[['Responder %', 'Non Responder %']]
            ks = ks.rename(columns={'Responder %':'Cummulative Responder %','Non Responder %':'Cummulative Non Responder %'})
            ks = ks.cumsum()
            ks['KS'] = ks['Cummulative Responder %'] - ks['Cummulative Non Responder %']
            df = pd.concat([df,ks], axis=1)
            
        if 'psi' in metrics:
            psi = df[['Population']].copy()
            psi['Baseline'] = [ 0.1 for _ in range(ntiles) ]
            psi['Population %'] = psi['Population'] / population_size
            psi['PSI'] = (psi['Population %'] - psi['Baseline']) * np.log(psi['Population %'] / psi['Baseline'])
            psi = psi.drop(['Population', 'Baseline'], axis=1)
            df = pd.concat([df,psi], axis=1)
        
        if 'cumsum' in metrics:
            cumsum = df[['Population','Responder']].copy()
            cumsum['Non Responder'] = cumsum['Population'] - cumsum['Responder']
            cumsum = cumsum.rename(columns={'Population':'Cummulative Population', 
                                            'Responder':'Cummulative Responder',
                                            'Non Responder':'Cummulative Non Responder'})
            cumsum = cumsum.cumsum()
            df = pd.concat([df,cumsum], axis=1)
            
        if 'odds' in metrics:
            odds = df[['Population','Responder']].copy()
            odds['Non Responder'] = odds['Population'] - odds['Responder']
            odds['Odds'] = odds['Responder'] / odds['Non Responder']
            df = pd.concat([df,odds['Odds']], axis=1)
            
        if 'lift' in metrics:
            lift = df[['Population','Responder']].copy()
            lift['Bad Rate'] = lift['Responder'] / lift['Population'] 
            lift['Lift'] = lift['Bad Rate'] / lift['Bad Rate'].mean()
            df = pd.concat([df,lift[['Bad Rate', 'Lift']]], axis=1)
            
        if 'separation' in metrics:
            sep = df[['Population','Responder']].copy()
            sep['Bad Rate'] = sep['Responder'] / sep['Population'] 
            sep['Separation'] = sep['Bad Rate'] / sep['Bad Rate'].min()
            df = pd.concat([df,sep[['Bad Rate', 'Separation']]], axis=1)   
        
        return df