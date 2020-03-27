import matplotlib.pyplot as pp

import numpy as np
import pandas as pd

import seaborn as sb

import yfinance as yf
import talib

from scipy.spatial.distance import correlation
from scipy.stats import t, norm

DEFAULT_RECO_METRICS = [
        'diversification_score', 
        '95% 1-period Student t CVaR', 
        'forwardPE', 
        'dividendYield',
        'beta', 
        'profitMargins', 
        'pegRatio',
        #'mfi',
        'governanceScore',
        'brownian_motion_score',
        #'rsi',
        'share_of_analyst_upgrades'
        #'earningsQuarterlyGrowth'
]

DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'MCD', 'DIS', 'GS']

class Recommender:
    symbols = None
    results = pd.DataFrame()

    def __init__(self, symbols:list=None):
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.relevant_metrics = DEFAULT_RECO_METRICS

        self.results = self._fetch_data(self.symbols)

    def get_all_metrics(self)->pd.DataFrame:
        """Return a dataframe with all calculated and collected metrics per symbol."""
        return self.results

    def get_rel_metrics(self)->pd.DataFrame:
        """Return a dataframe with all the metrics per symbol that were considered for the reco."""
        return self.results.sort_values('reco_score', ascending=False)[self.relevant_metrics+['reco_score']]

    def get_portfolio_summary(self, top_n =25)->pd.DataFrame:
        """
        Return ranked, recommended portfolios.

        Arguments: 
            top_n - number of symbols to be considered for the portfolio simulations

        Returns:
            Dataframe with all the portfolio combinations, sum of reco score and sharpe ratio

        """
        winners = self.results['reco_score'].sort_values(ascending=False).head(top_n).index.values
        df_w=self._get_history(winners, start='2007-01-01')

        portfolios = self._opt_portfolio(df_w['Adj Close'], df_w['Adj Close'].columns.unique(), iterations = 5000)
        dd=portfolios.head(50).T.join(self.results['reco_score']).sort_values('reco_score', ascending = False)

        df_d=pd.DataFrame(np.multiply((dd.values > 0).astype(np.int).T, dd['reco_score'].values)).iloc[:-1].dropna(axis=1)
        df_d.columns=dd.index[:-1]
        df_d.index=df_d.index+1

        df_d.sum(axis=1)
        return dd.append(pd.Series(df_d.sum(axis=1), name='sum_reco_score'))

    def plot_metrics(self, pairs:list=[('profitMargins', 'rsi')]):
        if len(pairs) == 1:
            metric_a, metric_b = pairs[0]
            df_t = self.results[[metric_a, metric_b]].dropna()

            _, ax = pp.subplots()
            
            x = df_t[metric_a].values
            y = df_t[metric_b].values

            ylim = (y.min()*0.8, y.max()*1.2)
            xlim = (0, x.max()*1.1)

            qx2, qx4 = np.percentile(x, 20), np.percentile(x, 80)
            qy2, qy4 = np.percentile(y, 20), np.percentile(y, 80)

            ax.axvline(qx2, linestyle='--', color='C2', linewidth=0.8)
            ax.axvline(qx4, linestyle='--', color='C2', linewidth=0.8)
            
            if metric_b == 'rsi':
                ax.fill_between(x=np.arange(xlim[0],xlim[1], 0.01), y1=ylim[0], y2=30, color='green', alpha=0.1)
                ax.fill_between(x=np.arange(xlim[0],xlim[1], 0.01), y1=30, y2=50, color='yellow', alpha=0.1)
                ax.fill_between(x=np.arange(xlim[0],xlim[1], 0.01), y1=50, y2=70, color='orange', alpha=0.15)
                ax.fill_between(x=np.arange(xlim[0],xlim[1], 0.01), y1=70, y2=ylim[1], color='red', alpha=0.2)
            else:
                ax.axhline(qy2, linestyle='--', color='C2', linewidth=0.8)
                ax.axhline(qy4, linestyle='--', color='C2', linewidth=0.8)

            ax.scatter(x=x, y=y, marker='o')
            for i, txt in enumerate(df_t.index):
                ax.annotate(txt, (x[i], y[i]))

            ax.set(xlim = xlim, ylim = ylim, xlabel=metric_a, ylabel=metric_b)
            
            return
        
        cols = np.ceil(np.sqrt(len(pairs))).astype(np.int)
        rows = np.floor(np.sqrt(len(pairs))).astype(np.int)
        
        if rows < 2:
            rows=2

        _, ax = pp.subplots(ncols=cols, nrows=rows, figsize=(16, 8))

        row, col = 0,0
        for metric_a, metric_b in pairs:
            if col >= cols:
                row += 1
                col = 0
                
            df_t = self.results[[metric_a, metric_b]].dropna()

            x = df_t[metric_a].values
            y = df_t[metric_b].values
            
            if x.max() < 1:
                x*=100
            
            if y.max() < 1:
                y*=100

            ylim = (y.min()*0.8, y.max()*1.2)
            xlim = (x.min()*0.8, x.max()*1.1)

            qx2, qx4 = np.percentile(x, 20), np.percentile(x, 80)
            qy2, qy4 = np.percentile(y, 20), np.percentile(y, 80)

            ax[col][col].axvline(qx2, linestyle='--', color='C2', linewidth=0.8)
            ax[row][col].axvline(qx4, linestyle='--', color='C2', linewidth=0.8)

            ax[row][col].axhline(qy2, linestyle='--', color='C2', linewidth=0.8)
            ax[row][col].axhline(qy4, linestyle='--', color='C2', linewidth=0.8)

            ax[row][col].scatter(x=x, y=y, marker='o')
            for i, txt in enumerate(df_t.index):
                ax[row][col].annotate(txt, (x[i], y[i]))

            ax[row][col].set(xlim = xlim, ylim = ylim, xlabel=metric_a, ylabel=metric_b)
            
            col+=1

    def _get_vars(self, data):
        def estimate_volatility(ret:np.array):
            returns = ret[np.isnan(ret)==False]
            dx = 0.0001  # resolution
            x = np.arange(returns.min(), returns.max(), dx)

            # N(x; mu, sig) best fit (finding: mu, stdev)
            mu_norm, sig_norm = norm.fit(returns)
            pdf = norm.pdf(x, mu_norm, sig_norm)

            # Student t best fit (finding: nu)
            parm = t.fit(returns)
            nu, mu_t, sig_t = parm
            nu = np.round(nu)
            pdf2 = t.pdf(x, nu, mu_t, sig_t)
            
            walks=self._get_brownian_motion_returns(nu, mu_t, sig_t, 50)
            med_walk, lower_walk, upper_walk = np.percentile(walks, 50, axis=1)[-1], np.percentile(walks, 20, axis=1)[-1], np.percentile(walks, 80, axis=1)[-1]

            h = 1
            alpha = 0.05  # significance level
            lev = 100*(1-alpha)
            xanu = t.ppf(alpha, nu)

            VaR_t = np.sqrt((nu-2)/nu) * t.ppf(1-alpha, nu)*sig_norm  - h*mu_norm
            CVaR_t = -1/alpha * (1-nu)**(-1) * (nu-2+xanu**2) * \
                            t.pdf(xanu, nu)*sig_norm  - h*mu_norm
            
            if (VaR_t < 0) or (CVaR_t <0):
                return {
                    'Sample Mean': np.nan,
                    'Sample StdDev':np.nan,
                    'nu': nu,
                    "%g%% %g-period Student t VaR" % (lev, h): np.nan,
                    "%g%% %g-period Student t CVaR" % (lev, h): np.nan,
                    "brownian_motion_score": 100*med_walk/(1+(upper_walk-lower_walk))**2,
                    'periods_included': len(returns)
                }

            return {
                'Sample Mean': mu_norm,
                'Sample StdDev': sig_norm,
                'nu': nu,
                "%g%% %g-period Student t VaR" % (lev, h): VaR_t*100,
                "%g%% %g-period Student t CVaR" % (lev, h): CVaR_t*100,
                "brownian_motion_score": 100*med_walk/(1+(upper_walk-lower_walk))**2,
                'periods_included': len(returns)
            }

        cols=data['Adj Close'].columns.unique()
        dta = pd.DataFrame()
        results=[]
        for c in cols:
            t1=data['Adj Close'][c].values
            t0 = np.roll(t1,shift=-1)
            ret = (t1[:-1]-t0[:-1])/t0[:-1]
            res=estimate_volatility(ret)
            res['symbol']=c
            results.append(res)

        return pd.DataFrame().from_dict(results)

    def _fetch_data(self, symbols:list)->pd.DataFrame:
        data = self._get_history(symbols)
            
        metrics = [
            '95% 1-period Student t VaR', 
            '95% 1-period Student t CVaR',
            'brownian_motion_score',
            'diversification_score',
            'updated'
        ]
        governance = [
            'socialScore', 
            'environmentScore', 
            'governanceScore',
            'highestControversy'
        ]
        fundamentals = [
            'country',
            'shortName',
            'sector',
            'industry',
            '52WeekChange',
            'earningsQuarterlyGrowth',
            'profitMargins',
            'forwardPE',
            'forwardEps',
            'priceToBook',
            'pegRatio',
            'dividendYield',
            'payoutRatio',
            'beta',
            'marketCap',
            'averageVolume',
            'fullTimeEmployees',
            'heldPercentInstitutions',
            'heldPercentInsiders',
            'shortRatio',
            'bookValue'
        ]

        results=self._get_vars(data)
        results['diversification_score']=np.nan
        results.set_index('symbol', inplace=True)
        
        rows = np.array([])
        symbols = results.index.values
        for s in symbols:
            results.at[s, 'diversification_score']=self._get_diversification_score(s, data, 'Adj Close')
                
            ticker= yf.Ticker(s)
            
            if ticker is None:
                continue
            
            row = dict()
            
            ts = ticker.history(period='52wk')

            row['rsi']=talib.RSI(ts['Close'])[-1]
            row['natr'] = talib.NATR(ts['High'], ts['Low'], ts['Close'], timeperiod=14)[-1]
            row['mfi'] = talib.MFI(ts['High'], ts['Low'], ts['Close'], ts['Volume'], timeperiod=14)[-1]
            
            try:
                gov = ticker.sustainability
            except:
                gov = None
            
            try:
                info = ticker.info
            except:
                info = None
            
            for v in governance:
                if gov is None:
                    row[v]=np.nan
                    continue
                    
                if v in ticker.sustainability.index:
                    row[v]=ticker.sustainability.loc[v][0]
                else:
                    row[v]=np.nan
            
            for v in fundamentals:
                if info is None:
                    row[v]=np.nan
                    continue
                    
                if v in ticker.info:
                    row[v]=ticker.info[v]
                else:
                    row[v]=np.nan

            row['symbol']=s
            
            row['share_of_analyst_upgrades']= self._get_analyst_upgrades(ticker)
                    
            rows=np.append(rows, row)
            
        results['updated']= pd.Timestamp.today().strftime('%Y-%m-%d')
        results=results[metrics]
        
        results = results.join(pd.DataFrame().from_records(rows).set_index('symbol'))

        results['dividendYield']=results['dividendYield'].fillna(0)

        results['reco_score'] = self._get_reco_score(results, self.relevant_metrics)
        
        return results

    ## static methods
    @staticmethod
    def _get_reco_score(df, relevant_metrics):
        df_ch=df[relevant_metrics]

        df_ch = df_ch / df_ch.mean(axis=0)

        df_ch['95% 1-period Student t CVaR']=1/df_ch['95% 1-period Student t CVaR']
        df_ch['beta']=1/df_ch['beta']
        #df_ch['mfi']=50/df_ch['mfi']
        df_ch['forwardPE']=1/df_ch['forwardPE']
        df_ch['pegRatio']=1/df_ch['pegRatio']
        df_ch['governanceScore']=1.5/df_ch['governanceScore']**2
        df_ch['profitMargins']=df_ch['profitMargins']**2

        df_ch.fillna(0, inplace=True)
        #df_ch['rsi']=50/df_ch['rsi']
        
        return df_ch.mean(axis=1)**2

    @staticmethod
    def _get_history(symbols, start='2000-01-01'):
        sym_str = ''
        for n, s in enumerate(symbols):
            if n > 0 and n < len(symbols):
                sym_str += ' '
            sym_str += s

        return yf.download(sym_str, start=start, interval='1wk')

    @staticmethod
    def _get_analyst_upgrades(ticker):
        rec = None
        
        try:
            rec=ticker.recommendations
        except:
            pass
        
        if rec is None:
            return np.nan
        
        upgrades = rec \
            .loc[rec.Action.isin(['up', 'down'])] \
            .tail(20) \
            .Action \
            .apply(
                lambda s: 1 if s == 'up' else 0
            ).mean()
        return upgrades

    @staticmethod
    def _get_diversification_score(symbol, data, metric):
        df=data[metric].dropna()
        
        rest = [c for c in df.columns.unique() if c != symbol]

        a = df[symbol].values
        dists = np.array([])
        for col in rest:
            dists = np.append(dists, correlation(a, df[col].values))

        return np.log(1+np.percentile(dists,50))

    @staticmethod
    def _get_brownian_motion_returns(nu, mu_t, sig_t, periods=52):
        T = periods
        S0 = 1.0

        dt = 0.1
        N = np.round(T/dt).astype(np.int)
        t_ = np.linspace(0, T, N)
        nu = np.round(nu)

        walks = []
        for n in range(250):
            W = np.random.standard_t(df=nu, size = N) 
            W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
            X = (mu_t-0.5*sig_t**2)*t_ + sig_t*W 
            S_t = S0*np.exp(X) ### geometric brownian motion ###
            walks.append(S_t)

        return np.array(walks).T

    @staticmethod
    def _opt_portfolio(df, symbols, budget=5000, iterations=1000):
        '''Calculates Share ratios for a number of randomly generated portfolios'''
        
        adj_closes = df[symbols]
        num_assets=len(symbols)
        returns = np.log(adj_closes / adj_closes.shift(1))
        
        port_returns = []
        port_vols = []
        sharpes = []
        weight_list = []
        
        for i in range (iterations):
            while True:
                a=np.random.choice([0,1], size = num_assets)
                if (a.sum() >= 3) and (a.sum() <= 6):
                    break
            weights=a/a.sum()
            
            weight_list.append(weights)
            
            port_return = np.sum(returns.mean() * weights) * 252
            port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe = port_return/port_vol
            
            port_returns.append(port_return)
            port_vols.append(port_vol)
            sharpes.append(sharpe)
        
        # Convert lists to arrays
        port_returns = np.array(port_returns)
        port_vols = np.array(port_vols)
        sharpes = np.array(sharpes)
        weight_list = np.array(weight_list)
        
        d={k:v*budget for k,v in zip(symbols, weight_list.T)}
        
        d['sharpes'] = sharpes
        res=pd.DataFrame(d).drop_duplicates()
        res.sort_values('sharpes', ascending=False, inplace=True)
        
        res.index=np.arange(1, len(res)+1)
        
        return res
