from functools import partial
from itertools import combinations
import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from segment import (
    semantic_similarity_sentence_transforms_all_combinations,
    semantic_similarity_openai_all_combinations,
    semantic_similarity_spacy, semantic_similarity_bert,
    semantic_similarity_openai,
    semantic_similarity_sentence_transformers
)
import plotly.express as px
import time
import numpy as np
import random

np.random.seed(0)
random.seed(0)

thisdir = pathlib.Path(__file__).parent.absolute()

SS_MODE = os.getenv('SS_MODE', 'sentence-transformers')


if SS_MODE == 'openai':
    semantic_similarity_all_combos = partial(semantic_similarity_openai_all_combinations, model='text-embedding-ada-002')
    semantic_similarity = partial(semantic_similarity_openai, model='text-embedding-ada-002')
else:
    semantic_similarity_all_combos = partial(semantic_similarity_sentence_transforms_all_combinations, model='all-MiniLM-L6-v2')
    semantic_similarity = partial(semantic_similarity_sentence_transformers, model='all-MiniLM-L6-v2')


def main():
    path = thisdir / '.results' / 'sentences-translated'
    savedir = thisdir.joinpath('.output')
    savedir.mkdir(exist_ok=True, parents=True)

    dfs = []
    for file in path.glob('*.csv'):
        df = pd.read_csv(file, index_col=0)
        df['model'] = file.stem
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna()

    # replace "he/she/it" (not case-sensitive) with "he"
    df['backwards'] = df['backwards'].str.replace(r'\b([Hh]e/[Ss]he/[Ii]t)\b', 'he', regex=True)

    # compute similarity metrics
    df['sim_simple'] = df.apply(lambda row: semantic_similarity(row['sentence'], row['simple']), axis=1)
    df['sim_comparator'] = df.apply(lambda row: semantic_similarity(row['sentence'], row['comparator']), axis=1)
    df['sim_backwards'] = df.apply(lambda row: semantic_similarity(row['sentence'], row['backwards']), axis=1)

    similarities = semantic_similarity_all_combos(df['sentence'].unique())
    # keep only upper triangle
    similarities = np.triu(similarities, k=1)
    # make 0s NaNs
    similarities[similarities == 0] = np.nan
    sim_mean = np.mean(similarities, where=~np.isnan(similarities))
    sim_std = np.std(similarities, where=~np.isnan(similarities))
    baseline_similarities = pd.Series(similarities.flatten())

    print(f'mean: {sim_mean}, std: {sim_std}')

    # plot baseline similarities as histogram with plotly
    fig = px.histogram(
        x=baseline_similarities,
        template='plotly_white',
        # make the bars gray
        color_discrete_sequence=['gray'],
    )
    # add x and y axis labels
    fig.update_xaxes(title_text='semantic similarity')
    fig.update_yaxes(title_text='count')
    
    fig.write_html(savedir / 'baseline_similarities.html')
    fig.write_image(savedir / 'baseline_similarities.pdf')
    time.sleep(1)
    fig.write_image(savedir / 'baseline_similarities.pdf') # do twice because of bug

    sim_metrics = ['sim_simple', 'sim_comparator', 'sim_backwards']
    df['translation_quality'] = df.apply(lambda row: row[sim_metrics].mean(), axis=1)
    # sort by translation_quality
    df = df.sort_values(by='translation_quality', ascending=False, ignore_index=True)

    threshold = sim_mean + 3*sim_std
    print(f'threshold: {threshold}')
    df['quality'] = 'Bad'
    df.loc[(df['sim_simple'] > threshold) & (df['sim_comparator'] > threshold) & (df['sim_backwards'] > threshold), 'quality'] = 'Good'
    df.loc[(df['sim_simple'] > threshold) & (df['sim_comparator'] < threshold) & (df['sim_backwards'] > threshold), 'quality'] = 'Good given vocab'

    # compute frequencies for each type and quality
    freqs = df.groupby(['model', 'type', 'quality']).size().unstack(fill_value=0)
    freqs = freqs.div(freqs.sum(axis=1), axis=0)

    freqs = freqs.sort_index(level=[0, 1], key=lambda x: x.map({
        'gpt-3.5-turbo': 0, 'gpt-4': 1,
        'subject-verb': 0, 'subject-verb-object': 1, 'two-verb': 2, 'two-clause': 3, 'complex': 4
    }))

    # write summary to table
    freqs.to_csv(savedir / 'translation_quality_summary.csv')
    freqs.to_html(savedir / 'translation_quality_summary.html')
    # convert to percentages
    print(freqs.to_latex(float_format=lambda x: f"{int(x*100)}\%"))

    # group tokens by model and sentence and sum them
    tokens_df = df.groupby(['model', 'sentence']).agg({
        'prompt_tokens': 'sum',
        'completion_tokens': 'sum',
    })
    # compute min, max, mean, std for prompt_tokens and completion_tokens
    tokens_summary = tokens_df.agg({
        'prompt_tokens': ['min', 'max', 'mean', 'std', 'sum'],
        'completion_tokens': ['min', 'max', 'mean', 'std', 'sum'],
    })
    tokens_summary.to_csv(savedir / 'translation_quality_tokens_summary.csv')
    tokens_summary.to_html(savedir / 'translation_quality_tokens_summary.html')
    print(tokens_summary.to_string())
    print(tokens_summary.to_latex(float_format=lambda x: f"{int(x)}"))

    token_costs = {
        'gpt-3.5-turbo': {
            'prompt_tokens': 0.5 / 1e6,
            'completion_tokens': 1.5 / 1e6,
        },
        'gpt-4': {
            'prompt_tokens': 30 / 1e6,
            'completion_tokens': 60 / 1e6,
        },
    }
    # compute token costs
    tokens_df = tokens_df.reset_index()
    tokens_df['prompt_cost'] = tokens_df.apply(lambda row: row['prompt_tokens'] * token_costs[row['model']]['prompt_tokens'], axis=1)
    tokens_df['completion_cost'] = tokens_df.apply(lambda row: row['completion_tokens'] * token_costs[row['model']]['completion_tokens'], axis=1)
    tokens_costs = tokens_df.groupby('model').agg({
        'prompt_cost': ['min', 'max', 'mean', 'std', 'sum'],
        'completion_cost': ['min', 'max', 'mean', 'std', 'sum'],
    })
    tokens_costs.to_csv(savedir / 'translation_quality_tokens_costs.csv')
    tokens_costs.to_html(savedir / 'translation_quality_tokens_costs.html')
    print(tokens_costs.to_string())

    # rename types
    df['type'] = df['type'].replace({
        'sv': 'subject-verb',
        'svo': 'subject-verb-object',
        '2v': 'two-verb',
        '2s': 'complex',
        '2c': 'two-clause',
    })

    print(df.groupby(by=["model", "type"]).agg({'translation_quality': ['mean', 'std']}).to_string())

    df = df.melt(
        id_vars=['model', 'sentence', 'type', 'simple', 'comparator', 'target', 'backwards'],
        value_vars=sim_metrics,
        var_name='similarity_metric',
        value_name='similarity'
    )

    rename_values = {
        'sim_simple': 'simple',
        'sim_comparator': 'comparator',
        'sim_backwards': 'backwards',
    }
    labels = {
        'similarity_metric': 'generated sentence',
        'type': 'sentence type',
    }

    symbol_map = {
        'gpt-3.5-turbo': 'diamond-tall',
        'gpt-4': 'diamond-wide',
    }
    marker_format = dict(size=15, opacity=0.8, line=dict(width=2, color='DarkSlateGrey'))

    fig = px.scatter(
        df.replace(rename_values),
        x='sentence',
        y='similarity',
        symbol='model',
        symbol_map=symbol_map,
        color='similarity_metric',
        facet_col='type',
        facet_col_wrap=2,
        template='plotly_white',
        custom_data=['sentence', 'simple', 'comparator', 'target', 'backwards'],
        labels=labels
    )
    fig.update_xaxes(matches=None)
    fig.for_each_xaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.update_traces(marker=marker_format, mode="markers", hovertemplate=None)
    fig.update_layout(
        autosize=False,
        width=1650,
        height=2500,
        hovermode="x unified",
    )
    def set_hoverdata(trace) -> str:
        if rename_values['sim_simple'] in trace.legendgroup:
            trace.update(hovertemplate="<br>" + "<br>".join([
                "simple: %{customdata[1]}",
                "similarity: %{y}",
            ]))
        elif rename_values['sim_comparator'] in trace.legendgroup:
            trace.update(hovertemplate="<br>" + "<br>".join([
                "comparator: %{customdata[2]}",
                "similarity: %{y}",
            ]))
        elif rename_values['sim_backwards'] in trace.legendgroup:
            trace.update(hovertemplate="<br>" + "<br>".join([
                "target: %{customdata[3]}",
                "backwards: %{customdata[4]}",
                "similarity: %{y}",
            ]))
    # add data to hover data
    fig.for_each_trace(set_hoverdata)
    fig.update_xaxes(tickangle=45)

    fig.add_hrect(
        y0=sim_mean - sim_std, y1=sim_mean + sim_std,
        line_width=0, fillcolor='gray',
        opacity=0.5, layer='below',
        row='all', col='all'
    )
    fig.add_hrect(
        y0=sim_mean - 2*sim_std, y1=sim_mean + 2*sim_std,
        line_width=0, fillcolor='gray',
        opacity=0.3, layer='below',
        row='all', col='all'
    )
    fig.add_hrect(
        y0=sim_mean - 3*sim_std, y1=sim_mean + 3*sim_std,
        line_width=0, fillcolor='gray',
        opacity=0.1, layer='below',
        row='all', col='all'
    )
    fig.update_yaxes(range=[0, 1.1])
    fig.write_html(savedir / 'translation_quality.html')
    fig.write_image(savedir / 'translation_quality.pdf')

    # generate same plot for each type separately
    for sim_type in df['type'].unique():
        df_type: pd.DataFrame = df[df['type'] == sim_type]
        fig = px.scatter(
            df_type.replace(rename_values),
            x='sentence',
            y='similarity',
            symbol='model',
            symbol_map=symbol_map,
            color='similarity_metric',
            facet_col='type',
            facet_col_wrap=2,
            template='plotly_white',
            custom_data=['sentence', 'simple', 'comparator', 'target', 'backwards'],
            labels=labels
        )
        fig.update_xaxes(matches=None)
        fig.for_each_xaxis(lambda yaxis: yaxis.update(showticklabels=True))
        fig.update_traces(marker=marker_format, mode="markers", hovertemplate=None)
        fig.update_layout(hovermode="x unified")

        # def set_hoverdata(trace) -> str:
        #     if rename_values['sim_simple'] in trace.legendgroup:
        #         trace.update(hovertemplate="<br>" + "<br>".join([
        #             "simple: %{customdata[1]}",
        #             "similarity: %{y}",
        #         ]))
        #     elif rename_values['sim_comparator'] in trace.legendgroup:
        #         trace.update(hovertemplate="<br>" + "<br>".join([
        #             "comparator: %{customdata[2]}",
        #             "similarity: %{y}",
        #         ]))
        #     elif rename_values['sim_backwards'] in trace.legendgroup:
        #         trace.update(hovertemplate="<br>" + "<br>".join([
        #             "target: %{customdata[3]}",
        #             "backwards: %{customdata[4]}",
        #             "similarity: %{y}",
        #         ]))

        # add data to hover data
        fig.for_each_trace(set_hoverdata)
        fig.update_xaxes(tickangle=45)

        fig.add_hrect(
            y0=sim_mean - sim_std, y1=sim_mean + sim_std,
            line_width=0, fillcolor='gray',
            opacity=0.5, layer='below',
            row='all', col='all'
        )
        fig.add_hrect(
            y0=sim_mean - 2*sim_std, y1=sim_mean + 2*sim_std,
            line_width=0, fillcolor='gray',
            opacity=0.3, layer='below',
            row='all', col='all'
        )
        fig.add_hrect(
            y0=sim_mean - 3*sim_std, y1=sim_mean + 3*sim_std,
            line_width=0, fillcolor='gray',
            opacity=0.1, layer='below',
            row='all', col='all'
        )

        fig.write_html(savedir / f'translation_quality_{sim_type}.html')
        # make figure narrower
        fig.update_layout(
            autosize=False,
            width=1000,
            height=500,
        )
        # set y-axis range
        fig.update_yaxes(range=[0, 1.1])
        fig.write_image(savedir / f'translation_quality_{sim_type}.pdf')


if __name__ == '__main__':
    main()

