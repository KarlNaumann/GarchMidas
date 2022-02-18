"""
Module to export a variety of items into usable latex ouput
-------------------------------------------------------
Currently implements only pandas DataFrame items
"""

__author__ = "Karl Naumann-Woleske"
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Karl Naumann-Woleske"

import pandas as pd


def latex_matrix(df, eq=None, precision=None, bracket_cols=[], save_file=None,
                 verbose=False) -> str:
    """ Export a dataframe matrix to a latex pmatrix

    Parameters
    ----------
    df : pd.DataFrame
    eq : str
    precision : int
    bracket_cols : list
    save_file : str
    verbose : bool

    Returns
    -------
    out : str
    """
    output = []
    if eq is not None:
        output.append('\\begin{equation}')
        output.append('\\hat{\\beta}_{%s}=' % eq)
    output.append('\\begin{pmatrix}')
    rows, cols = df.shape
    for i in range(rows):
        newrow = ''
        for j in range(cols):
            val = df.iloc[i, j]
            if precision is not None: val = round(val, precision)
            val = str(val)
            if j in bracket_cols: val = '(' + val + ')'
            newrow += f'{val} & '
        newrow = newrow[:-2]  # delete last "& "
        if i < rows - 1: newrow += r'\\'
        output.append(newrow)
    output.append('\\end{pmatrix}')
    if eq is not None: output.append('\\end{equation}')
    if save_file is not None:
        with open(save_file, 'w') as f:
            f.write('\n'.join(output))
    if verbose: print('\n'.join(output))
    return '\n'.join(output)


def latex_table(df: pd.DataFrame, title: str, label: str, indexTitle: str = '',
                save: str = True, precision: int = None, index: bool = True,
                verbose: bool = False):
    """ Export a dataframe as latex table

    Parameters
    ----------
    df: pd.DataFrame
    title: str
    label: str
    indexTitle: str = ''
    save: str = True
    precision: int = None
    index: bool = True
    verbose: bool = False

    Returns
    -------
    out : str
    """
    output = []
    # Basic beginning structure of the LaTeX table
    output.append('\\begin{table}[H]')
    output.append('\\centering')
    output.append('\\caption{%s} \\label{%s}' % (title, label[:-4]))
    # Formatting the alignment of the LaTeX matrix
    rows, cols = df.shape
    r = ''.join(['r' for x in range(len(df.columns))])
    if index:
        if type(df.index) == pd.core.indexes.multi.MultiIndex:
            r = 'l' * len(df.index.levels) + r
        else:
            r = 'l' + r
    output.append('\\begin{tabular}{%s}' % (r))
    # Header row
    header = ''
    if index:
        if type(df.index) == pd.core.indexes.multi.MultiIndex:
            for name in df.index.names:
                header += '{%s} &' % name
        else:
            header += '{%s} &' % indexTitle
    for col in df.columns:
        header += '%s &' % col
    # Delete the last '&' and put '\\'
    output.append(header[:-2] + '\\\\')
    output.append('\\midrule')
    # Body of the table
    priorrow = ''
    for i in range(rows):
        newrow = ''
        if index:
            if type(df.index) == pd.core.indexes.multi.MultiIndex:
                for j in range(len(df.index[i])):
                    if i > 0 and df.index[i][j] == df.index[i - 1][j]:
                        newrow += '\\textbf{} & '
                    else:
                        newrow += '\\textbf{%s} & ' % df.index[i][j]
            else:
                newrow += '\\textbf{%s} & ' % df.index[i]
        for j in range(cols):
            val = df.iloc[i, j]
            if precision is not None: val = round(val, precision)
            newrow += '{} & '.format(val)
        output.append(newrow[:-2] + '\\\\')
        priorrow = newrow
    # End of table format
    output.append('\\bottomrule')
    output.append('\\end{tabular}')
    output.append('\\end{table}')
    # write the table to a file
    if save:
        with open(label, 'w') as f:
            f.write('\n'.join(output))
            f.close()
    if verbose: print('\n'.join(output), '\n')
    # return string of the latex output
    return output


def regression_table(df_coef, df_se, title, label, df_pval=None, indexTitle='',
                     save=True, precision=None, index=True, verbose=False):
    """ Export a table for a regression with coefficients and standard errors

    Parameters
    ----------
    df_coef : pd.DataFrame
    df_se : pd.DataFrame
    title : str
    label : str
    df_pval : pd.DataFrame
    indexTitle : str
    save : str
    precision : int
    index : bool
    verbose : bool

    Returns
    -------
    out : str
    """
    output = []
    # Basic beginning structure of the LaTeX table
    output.append('\\begin{table}[H]')
    output.append('\\centering')
    output.append('\\caption{%s} \\label{%s}' % (title, label[:-4]))
    # Formatting the alignment of the LaTeX matrix
    rows, cols = df_coef.shape
    r = ''.join(['r' for x in range(len(df_coef.columns))])
    if index:
        if type(df_coef.index) == pd.core.indexes.multi.MultiIndex:
            r = 'l' * len(df_coef.index.levels) + r
        else:
            r = 'l' + r
    output.append('\\begin{tabular}{%s}' % (r))
    # Header row
    header = ''
    if index:
        if type(df_coef.index) == pd.core.indexes.multi.MultiIndex:
            for name in df_coef.index.names:
                header += '{%s} &' % name
        else:
            header += '{%s} &' % indexTitle
    for col in df_coef.columns:
        header += '%s &' % col
    # Delete the last '&' and put '\\'
    output.append(header[:-2] + '\\\\')
    output.append('\\hline')

    # Body of the table
    priorrow = ''
    prior_c = False
    for i in range(rows):
        newrow = ''
        if index:
            if not prior_c:
                if type(df_coef.index) == pd.core.indexes.multi.MultiIndex:
                    for j in range(len(df_coef.index[i])):
                        if i > 0 and df_coef.index[i][j] == df_coef.index[i - 1][j]:
                            newrow += '\\textbf{} & '
                        else:
                            newrow += '\\textbf{%s} & ' % df_coef.index[i][j]
                else:
                    newrow += '\\textbf{%s} & ' % df_coef.index[i]
            else:
                if type(df_coef.index) == pd.core.indexes.multi.MultiIndex:
                    for j in range(len(df_coef.index[i])): newrow += '\\textbf{} & '
                else:
                    newrow += '\\textbf{} & '

        if prior_c:
            for j in range(df_se.columns.shape[0]):
                val = df_coef.iloc[i, j]
                if precision is not None: val = round(val, precision)
                newrow += '\\scriptsize{(%s)} & ' % val
                prior_c = False
        else:
            for j in range(cols):
                val = df_coef.iloc[i, j]
                if precision is not None: val = round(val, precision)
                if df_pval is not None:
                    pval = df_pval.iloc[i, j]
                    if pval <= 0.01:
                        newrow += '%s\\textsuperscript\{***\} & ' % val
                    elif pval <= 0.05:
                        newrow += '%s\\textsuperscript\{**\} & ' % val
                    elif pval <= 0.1:
                        newrow += '%s\\textsuperscript\{*\} & ' % val
                else:
                    newrow += '{} & '.format(val)
            prior_c = True
        output.append(newrow[:-2] + '\\\\')
        priorrow = newrow
    # End of table format
    output.append('\\bottomrule')
    output.append('\\end{tabular}')
    output.append('\\end{table}')
    # write the table to a file
    if save:
        with open(label, 'w') as f:
            f.write('\n'.join(output))
            f.close()
    if verbose: print('\n'.join(output), '\n')
    # return string of the latex output
