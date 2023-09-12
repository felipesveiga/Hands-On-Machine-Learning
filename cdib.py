from numpy import ndarray
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV
from sympy import binomial
from typing import Dict, Optional

def X_y(filepath:str=None):
    '''
        Lê um arquivo e retorna as variáveis independentes e dependentes.

        Parâmetro
        ---------
        `filepath`: str
            Path do arquivo.

        Retorna
        -------
        Uma tupla contendo o DataFrame com variáveis independentes e depententes do dataset.
    '''
    data = pd.read_csv(filepath)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    return X, y

class Extract_CV:
    '''
        Responsável por, em uma única chamada, ler um .csv de tarefas supervisionadas e rodar uma bateria de validações
        cruzadas. 

        Parâmetros
        ----------
        `filepath`: str 
            Caminho do .csv
        `model`: 
            O modelo a ser usado.
        `scoring`: str
            Métrica de score.
        `cv`: int 
            Número de folds da validação cruzada.
        `param_grid`: dict
            Opcional: Dicionário de hiperparâmetros. Use apenas quando quiser fazer uma GridSearchCV.
     '''
    @staticmethod
    def __only_train_test(d:Dict[str, ndarray], keep:list=['mean_train_score', 'mean_test_score', 'train_score', 'test_score'])->dict:
        '''
            Mantém apenas os scores médios de treino e teste de um `cv_results_`.

            Parâmetros
            ----------
            `d`: Dicionário de `cv_results`.
            `keep`: Lista com os termos a serem mantidos no dicionário.

            Retorna
            -------
            O dicionário `cv_results` com apenas as chaves desejadas.
        '''
        return {key:d[key].mean() for key in keep if key in d.keys()}

    def __init__(self, filepath:str, model, scoring:str, cv:int, param_grid:Optional[dict]=None):
        self.filepath = filepath
        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.param_grid = param_grid

    def cross_validation(self)->dict:
        '''
            Performa a validação cruzada de um modelo, retornando os scores de treino e teste médios.

            Retorna
            -------
            Um dicionário com os scores de treino e teste médios.

        '''
        X, y = X_y(self.filepath)

        # Fazendo GridSearch apenas se `param_grid` for especificado.
        if self.param_grid:
            cv = GridSearchCV(self.model, param_grid=self.param_grid, scoring=self.scoring, cv=self.cv, return_train_score=True).fit(X,y).cv_results_

        # Caso o contrário, fazer apenas validação cruzada.
        else:
            cv = cross_validate(self.model, X, y, scoring=self.scoring, cv=self.cv, return_train_score=True)
        output = self.__only_train_test(cv)
        return output
    
def binomial_pmf(p:float, x:int, n:int)->float:
    '''
        Calcula a PMF de uma Distribuição Binomial.

        Use-a em situações de amostragens independentes (com reposição).

        Parâmetros
        ----------
        `p`: Probabilidade do sucesso.
        `x`: Quantidade de sucessos esperada.
        `n`: Quantidade de amostragens. 

        Retorna
        -------
        A probabilidade de sucessos P(X=x).
    '''
    return binomial(n, x) * p ** x * (1-p) ** (n-x)

def hypergeom_pmf(x:int, a:int, N:int, n:int)->float:
    '''
        Calcula a PMF de uma Distribuição Hipergeométrica.

        Use-a em situações de amostragens dependentes (sem reposição).

        Parâmetros
        ----------
        `x`: Quantidade de sucessos esperada.
        `a`: Número de pessoas na população com a característica desejada.
        `N`: Tamanho da população.
        `n`: Quantidade de amostragens. 

        Retorna
        -------
        A probabilidade de sucessos P(X=x).
    '''
    return binomial(a,x) * binomial(N-a,n-x) / binomial(N,n)