"""
Shared DataFrame helpers used across all pipeline merge scripts.
"""

import os
import pandas as pd


def combine_csvs(folder: str) -> pd.DataFrame:
    """Read all CSVs in a folder and concatenate into one DataFrame."""
    if not os.path.exists(folder):
        print(f'Folder not found: {folder} — subtopics will be empty.')
        return pd.DataFrame()
    dfs = [pd.read_csv(os.path.join(folder, f))
           for f in os.listdir(folder) if f.endswith('.csv')]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def explode_column(df: pd.DataFrame, id_col: str, sentiment_col: str,
                   topic_col: str, value_col: str, type_label: str) -> pd.DataFrame:
    """
    Explode a comma-separated column (pain_points or moments_of_delight) into one row each.
    Assigns a unique pp_mod_id per row used for joining subtopic mappings.
    """
    result = df[[id_col, sentiment_col, topic_col, value_col]].dropna(subset=[value_col]).copy()
    result[value_col] = result[value_col].astype(str).str.split(',')
    result = result.explode(value_col)
    result[value_col] = result[value_col].str.lower().str.strip()
    result = result[result[value_col] != ''].dropna(subset=[value_col])
    result.reset_index(drop=True, inplace=True)
    result['pp_mod_id'] = result[id_col].astype(str) + '-' + result.index.astype(str)
    result = result.rename(columns={id_col: 'Review_No', value_col: 'pp_mod'})
    result['type'] = type_label
    return result


_GENERIC_MODS = {'poor experience with', 'great experience with'}


def fix_subtopic_format(val) -> str:
    """
    Convert 'modifier-Root Subtopic' → 'Modifier Root Subtopic'.
    Splits only on the FIRST hyphen, preserving compound words like Check-in, In-flight.
    Strips generic fallback modifiers (Poor Experience with, Great Experience with).
    """
    if pd.isna(val) or str(val).strip() == '':
        return val
    s = str(val).strip()
    if '-' not in s:
        return s
    left, _, right = s.partition('-')
    left, right = left.strip(), right.strip()
    if left.lower() in _GENERIC_MODS:
        return right
    return left.title() + ' ' + right


def build_subtopic_lists(taxonomy_df: pd.DataFrame, entity_type: str):
    """
    Build pain-point and delight subtopic DataFrames from the master taxonomy.
    Filters out generic fallback modifiers and formats as 'Modifier Root'.
    Returns (pains_df, delights_df).
    """
    ts = taxonomy_df[taxonomy_df['type'] == entity_type].copy()
    GENERIC = {'poor experience with', 'great experience with'}

    def _build(mod_col):
        df = ts[['topic', 'root_subtopic', mod_col]].copy()
        df[mod_col] = df[mod_col].str.split(',')
        df = df.explode(mod_col)
        df[mod_col] = df[mod_col].str.strip()
        df = df.dropna(subset=[mod_col])
        df = df[~df[mod_col].str.lower().isin(GENERIC)]
        df['final_subtopic'] = df[mod_col].str.title() + ' ' + df['root_subtopic']
        return df

    return _build('negative_mods'), _build('positive_mods')
