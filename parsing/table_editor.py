import pandas as pd
import time
from IPython.display import display
import kickstarter_parser
import ast
import random
import os


def main():
    """
    Main function that orchestrates the scraping and data processing workflow.
    Handles loading Kickstarter data, processing Video Games projects,
    and saving results with scraped descriptions and media URLs.
    """
    # Load main Kickstarter dataset
    print('Начало работы:')
    df = pd.read_csv("kickstarter_datasets/Kickstarter_2019-07-18T03_20_05_009Z/Kickstarter.csv")
    
    # Data preparation steps
    df = convert_datetime_coumns(df)
    df = add_categories(df)
    
    # Filter for Video Games projects only
    video_games_df = df[df['category_name'] == "Video Games"].copy()
    
    columns_to_drop = ['category', 'country', 'currency_symbol', 'urls', 'source_url']
    video_games_df = video_games_df.drop(columns_to_drop, axis=1)
    
    # Initialize new columns for scraped data
    video_games_df['description'] = None 
    video_games_df['images'] = None
    video_games_df['videos'] = None
    
    # Load or create results dataframe
    if os.path.exists("combined_df.csv"):
        try:
            result_df = pd.read_csv("combined_df.csv")
            if result_df.empty:
                result_df = pd.DataFrame(columns=video_games_df.columns.tolist())
        except (pd.errors.EmptyDataError, FileNotFoundError):
            result_df = pd.DataFrame(columns=video_games_df.columns.tolist())
    else:
        result_df = pd.DataFrame(columns=video_games_df.columns.tolist())
    
    
    for index, row in video_games_df.iterrows():
        url = row['project_url']
        
        if not result_df.empty and url in result_df['project_url'].values:
            print(f'Проект {index} уже обработан, пропускаем...')
            continue
        
        try:
            print(f'\nОбработка проекта {index}...')
            # Use parser to get project description and media
            result = kickstarter_parser.get_description(url)
            
            if not result:
                print(f'Парсер вернул некорректные данные для проекта {index}')
                continue
            
            # Create new row with scraped data  
            new_row = row.copy()
            new_row['description'] = result.get('description', '')
            new_row['images'] = str(result['media'].get('images', []))  
            new_row['videos'] = str(result['media'].get('videos', []))
            
            # Append to results dataframe
            result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save progress after each iteration
            result_df.to_csv("combined_df.csv.csv", index=False)
            print(f'Проект {index} успешно добавлен в combined_df.csv.csv')
            
        except Exception as e:
            print(f'Ошибка при обработке проекта {index}: {str(e)}')
            with open("error_log.txt", "a") as f:
                f.write(f"Проект {index}, URL: {url}, Ошибка: {str(e)}\n")
            continue
            
        time.sleep(random.randint(2, 4))
    
    print("\nВсе данные успешно сохранены в combined_df.csv.csv")


def convert_datetime_coumns(df):
    """
    Converts Unix timestamp columns to datetime objects.
    
    Args:
        df: Input DataFrame containing timestamp columns
        
    Returns:
        DataFrame with converted datetime columns
    """
    cols_to_convert = ['created_at', 'deadline', 'launched_at', 'state_changed_at']
    for c in cols_to_convert:
        df[c] = pd.to_datetime(df[c], origin='unix', unit='s')
    print(f"The dataset contains projects added to Kickstarter between {min(df.created_at).strftime('%d %B %Y')} and {max(df.created_at).strftime('%d %B %Y')}.'\n\n'")
    return df

def df_columns_info(df):
    """
    Provides detailed analysis and statistics about DataFrame columns.
    Primarily used for exploratory data analysis.
    
    Args:
        df: Input DataFrame to analyze
    """
    pd.set_option('display.max_rows', None) 
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("=== Полная статистика по категориям ===")
    print(df['category_name'].describe(include='all'))

    print("\n=== Все уникальные категории и их количество ===")
    with pd.option_context('display.max_rows', None):
        print(df['category_name'].value_counts(dropna=False)) 

    print("\n=== Доля каждой категории ===")
    with pd.option_context('display.max_rows', None):
        print(df['category_name'].value_counts(normalize=True, dropna=False))

    print("\n=== Анализ пропущенных значений ===")
    print("Общее количество пропущенных:", df['category_name'].isnull().sum())
    print("\nСтроки с пропущенными категориями:")
    print(df[df['category_name'].isnull()])

    print("\n=== Полный алфавитный список всех категорий ===")
    print(sorted(df['category_name'].dropna().unique()))
    return

def add_categories(df):
    """
    Extracts category names and project URLs from nested structures.
    
    Args:
        df: Input DataFrame with raw category and URL data
        
    Returns:
        DataFrame with extracted category names and project URLs
    """
    df['category_name'] = df['category'].apply(lambda x: ast.literal_eval(x).get('name') if pd.notnull(x) else None)
    df['urls'] = df['urls'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['project_url'] = df['urls'].apply(lambda x: x.get('web', {}).get('project') if isinstance(x, dict) else None)
    return df

if __name__ == "__main__":
    main()
    
