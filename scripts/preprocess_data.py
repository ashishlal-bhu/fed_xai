import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import gc
import logging
from datetime import datetime
from typing import Dict, Tuple, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('preprocessing')

def print_memory_usage():
    """Print current memory usage."""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024**2:.2f} MB")

def optimize_df(df):
    """Optimize dataframe memory usage with NA handling."""
    for col in df.columns:
        has_na = df[col].isna().any()
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            if has_na:
                df[col] = df[col].astype('float32')
            else:
                df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
    return df

def load_large_table(filename, sampled_ids, usecols=None, chunksize=10000000):
    """Load large tables in chunks with filtering."""
    logger.info(f"Loading {filename} in chunks")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    file_path = os.path.join(base_dir, filename)
    
    dtype_dict = {
        'SUBJECT_ID': 'float32',
        'HADM_ID': 'float32',
        'ICUSTAY_ID': 'float32',
        'ITEMID': 'category',
        'VALUENUM': 'float32',
        'DOB': 'str',  # Added DOB handling
        'ADMITTIME': 'str',  # Added ADMITTIME handling
        'DISCHTIME': 'str'   # Added DISCHTIME handling
    }
    
    chunks = []
    total_rows = 0
    
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, 
                                                dtype=dtype_dict,
                                                usecols=usecols,
                                                chunksize=chunksize,
                                                na_values=['', 'nan', 'NaN', 'NULL'],
                                                low_memory=False)):  # Added low_memory=False
        # Create mask for filtering
        mask = chunk['HADM_ID'].isin(sampled_ids)
        
        # Filter rows and create a copy
        filtered_chunk = chunk[mask].copy()
        
        if len(filtered_chunk) > 0:
            # Convert IDs to int32 after filtering using .loc
            for col in ['SUBJECT_ID', 'HADM_ID']:
                if col in filtered_chunk.columns:
                    filtered_chunk.loc[:, col] = filtered_chunk[col].fillna(-1).astype('int32')
            
            chunks.append(filtered_chunk)
            total_rows += len(filtered_chunk)
            
            if chunk_num % 10 == 0:
                logger.info(f"Processed {chunk_num} chunks, current total rows: {total_rows}")
                print_memory_usage()
        
        # Clear memory
        del chunk, filtered_chunk
        gc.collect()
    
    if not chunks:
        return pd.DataFrame(columns=usecols if usecols else [])
    
    result = pd.concat(chunks, ignore_index=True)
    logger.info(f"Finished loading {filename}. Final shape: {result.shape}")
    return optimize_df(result)

def load_and_sample_data(sample_fraction=0.1):
    """Load and subsample data with proper table relationships."""
    try:
        logger.info(f"Loading data with {sample_fraction*100}% sampling")
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        
        # Load ADMISSIONS table
        logger.info("Loading ADMISSIONS.csv")
        admissions = pd.read_csv(
            os.path.join(base_dir, 'ADMISSIONS.csv'),
            usecols=['SUBJECT_ID', 'HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 
                    'ADMITTIME', 'DISCHTIME'],
            dtype={
                'SUBJECT_ID': 'float32',
                'HADM_ID': 'float32',
                'HOSPITAL_EXPIRE_FLAG': 'float32',
                'ADMITTIME': 'str',
                'DISCHTIME': 'str'
            },
            na_values=['', 'nan', 'NaN', 'NULL'],
            low_memory=False
        )
        
        # Load PATIENTS table
        logger.info("Loading PATIENTS.csv")
        patients = pd.read_csv(
            os.path.join(base_dir, 'PATIENTS.csv'),
            usecols=['SUBJECT_ID', 'DOB'],
            dtype={
                'SUBJECT_ID': 'float32',
                'DOB': 'str'
            },
            na_values=['', 'nan', 'NaN', 'NULL'],
            low_memory=False
        )
        
        # Merge ADMISSIONS with PATIENTS to get DOB
        logger.info("Merging ADMISSIONS and PATIENTS")
        admissions = admissions.merge(patients[['SUBJECT_ID', 'DOB']], 
                                    on='SUBJECT_ID', 
                                    how='left')
        
        # Handle missing values and convert types
        admissions = admissions.dropna(subset=['HADM_ID', 'SUBJECT_ID'])
        for col in ['SUBJECT_ID', 'HADM_ID']:
            admissions.loc[:, col] = admissions[col].fillna(-1).astype('int32')
        admissions.loc[:, 'HOSPITAL_EXPIRE_FLAG'] = admissions['HOSPITAL_EXPIRE_FLAG'].fillna(0).astype('int8')
        
        # Convert datetime columns
        datetime_cols = ['ADMITTIME', 'DISCHTIME', 'DOB']
        for col in datetime_cols:
            admissions.loc[:, col] = pd.to_datetime(admissions[col], errors='coerce')
        
        # Subsample
        pos_samples = admissions[admissions['HOSPITAL_EXPIRE_FLAG'] == 1]
        neg_samples = admissions[admissions['HOSPITAL_EXPIRE_FLAG'] == 0]
        
        pos_sampled = pos_samples.sample(frac=sample_fraction, random_state=42)
        neg_sampled = neg_samples.sample(frac=sample_fraction, random_state=42)
        
        sampled_admissions = pd.concat([pos_sampled, neg_sampled])
        sampled_hadm_ids = set(sampled_admissions['HADM_ID'])
        
        logger.info(f"Original samples: {len(admissions)}, Sampled: {len(sampled_admissions)}")
        logger.info(f"Original mortality rate: {len(pos_samples)/len(admissions):.3f}")
        logger.info(f"Sampled mortality rate: {len(pos_sampled)/len(sampled_admissions):.3f}")
        
        # Load other tables
        data = {
            'base_df': sampled_admissions,
            'labevents': load_large_table(
                'LABEVENTS.csv',
                sampled_hadm_ids,
                usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'VALUENUM']
            ),
            'chartevents': load_large_table(
                'CHARTEVENTS.csv',
                sampled_hadm_ids,
                usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'VALUENUM']
            ),
            'procedures': load_large_table(
                'PROCEDURES_ICD.csv',
                sampled_hadm_ids
            ),
            'diagnoses': load_large_table(
                'DIAGNOSES_ICD.csv',
                sampled_hadm_ids
            ),
            'prescriptions': load_large_table(
                'PRESCRIPTIONS.csv',
                sampled_hadm_ids
            )
        }
        
        # Verify DOB is present
        if 'DOB' not in data['base_df'].columns:
            raise ValueError("DOB column is missing after data loading")
        
        logger.info("Data loading completed successfully")
        return data
        
    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}")
        raise

def extract_features(df, data_dict):
    """Extract all features from the data with safe datetime handling."""
    logger.info("Extracting features")
    
    try:

        # Ensure datetime columns are properly converted
        datetime_cols = ['ADMITTIME', 'DISCHTIME', 'DOB']
        for col in datetime_cols:
            if col in df.columns:
                # Check if column is already datetime
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    logger.info(f"Converting {col} to datetime")
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Check if conversion was successful
        for col in datetime_cols:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.warning(f"Failed to convert {col} to datetime. Sample values: {df[col].head()}")

        # Calculate age more safely
        df['AGE'] = ((df['ADMITTIME'].dt.year - df['DOB'].dt.year) -
                    ((df['ADMITTIME'].dt.month < df['DOB'].dt.month) |
                     ((df['ADMITTIME'].dt.month == df['DOB'].dt.month) &
                      (df['ADMITTIME'].dt.day < df['DOB'].dt.day)))).astype('float32')
        
        # Calculate LOS in days more safely
        df['LOS'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds().div(86400).astype('float32')
        
        # Handle invalid values
        df.loc[df['AGE'] < 0, 'AGE'] = np.nan
        df.loc[df['AGE'] > 120, 'AGE'] = np.nan
        df['AGE'].fillna(df['AGE'].median(), inplace=True)
        df['LOS'].fillna(df['LOS'].median(), inplace=True)
        
        # Lab values
        lab_items = {
            '50912': 'CREATININE',
            '50971': 'POTASSIUM',
            '50983': 'SODIUM',
            '50902': 'CHLORIDE',
            '50882': 'BICARBONATE',
            '51222': 'HEMOGLOBIN',
            '51301': 'WBC',
            '51265': 'PLATELET',
            '50931': 'GLUCOSE'
        }
        
        logger.info("Processing lab features")
        for itemid, name in lab_items.items():
            # Process one lab at a time to save memory
            temp = data_dict['labevents'][data_dict['labevents']['ITEMID'] == itemid]
            temp = temp.groupby('HADM_ID')['VALUENUM'].first().astype('float32')
            df = df.merge(temp.to_frame(name=name), 
                         on='HADM_ID', 
                         how='left')
            gc.collect()
            
        # Vital signs
        vitals = {
            '220045': 'HEART_RATE',
            '220050': 'SYSTOLIC_BP',
            '220051': 'DIASTOLIC_BP',
            '220179': 'TEMPERATURE',
            '220210': 'RESPIRATORY_RATE',
            '220277': 'SPO2'
        }
        
        logger.info("Processing vital signs")
        for itemid, name in vitals.items():
            temp = data_dict['chartevents'][data_dict['chartevents']['ITEMID'] == itemid]
            temp = temp.groupby('HADM_ID')['VALUENUM'].first().astype('float32')
            df = df.merge(temp.to_frame(name=name),
                         on='HADM_ID',
                         how='left')
            gc.collect()
            
        # Process procedures
        logger.info("Processing procedures")
        if 'procedures' in data_dict and not data_dict['procedures'].empty:
            # Check if ICD9_CODE column exists
            if 'ICD9_CODE' in data_dict['procedures'].columns:
                # Convert ICD9_CODE to string if it's not already
                logger.info("Converting ICD9_CODE to string type")
                data_dict['procedures']['ICD9_CODE'] = data_dict['procedures']['ICD9_CODE'].astype(str)
                
                # Now you can safely use string operations
                data_dict['procedures']['PROC_CAT'] = data_dict['procedures']['ICD9_CODE'].str[:2]
                
                # Continue with the rest of the procedure processing...
                proc_counts = data_dict['procedures'].groupby('HADM_ID').size()
                df = df.merge(proc_counts.to_frame(name='PROCEDURE_COUNT'),
                             on='HADM_ID',
                             how='left')
                
                # Top procedure categories
                top_procs = data_dict['procedures']['PROC_CAT'].value_counts().nlargest(10).index
                
                for proc in top_procs:
                    proc_df = data_dict['procedures'][data_dict['procedures']['PROC_CAT'] == proc]
                    proc_df = proc_df.groupby('HADM_ID').size().to_frame(name=f'PROC_{proc}')
                    df = df.merge(proc_df,
                                 on='HADM_ID',
                                 how='left')
                    gc.collect()
            else:
                logger.warning("ICD9_CODE column not found in procedures data")
                df['PROCEDURE_COUNT'] = 0
        else:
            logger.warning("Procedures data is empty or missing")
            df['PROCEDURE_COUNT'] = 0
            
        # Process diagnoses
        logger.info("Processing diagnoses")

        if 'diagnoses' in data_dict and not data_dict['diagnoses'].empty:
        # Check if ICD9_CODE column exists
            if 'ICD9_CODE' in data_dict['diagnoses'].columns:
                # Convert ICD9_CODE to string
                logger.info("Converting diagnoses ICD9_CODE to string type")
                data_dict['diagnoses']['ICD9_CODE'] = data_dict['diagnoses']['ICD9_CODE'].astype(str)
                
                # Continue with diagnoses processing...
                diag_counts = data_dict['diagnoses'].groupby('HADM_ID').size()
                df = df.merge(diag_counts.to_frame(name='DIAGNOSIS_COUNT'),
                            on='HADM_ID',
                            how='left')
            
                # Charlson comorbidities
                charlson_codes = {
                    'MI': ['410', '412'],
                    'CHF': ['428'],
                    'Diabetes': ['250']
                }
                
                for condition, codes in charlson_codes.items():
                    condition_mask = data_dict['diagnoses']['ICD9_CODE'].str.startswith(tuple(codes), na=False)
                    condition_df = data_dict['diagnoses'][condition_mask]
                    condition_df = condition_df.groupby('HADM_ID').size().to_frame(name=f'HAS_{condition}')
                    df = df.merge(condition_df,
                                on='HADM_ID',
                                how='left')
                    gc.collect()
            else:
                logger.warning("ICD9_CODE column not found in diagnoses data")
                df['DIAGNOSIS_COUNT'] = 0
        else:
            logger.warning("Diagnoses data is empty or missing")
            df['DIAGNOSIS_COUNT'] = 0
            
        # Fill NAs with 0 for count features
        count_cols = [col for col in df.columns if col.startswith(('PROC_', 'HAS_', 'DIAGNOSIS_', 'PROCEDURE_'))]
        df[count_cols] = df[count_cols].fillna(0)
        
        # Fill NAs with median for numeric features
        numeric_cols = ['AGE', 'LOS'] + list(lab_items.values()) + list(vitals.values())
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Convert all features to float32
        feature_cols = numeric_cols + count_cols
        df[feature_cols] = df[feature_cols].astype('float32')
        
        logger.info(f"Feature extraction completed. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise

def preprocess_data(task="mortality", sample_fraction=0.1):
    """Main preprocessing function."""
    try:
        print_memory_usage()
        logger.info(f"Starting preprocessing for task: {task}")
        
        # Load data with subsampling
        data = load_and_sample_data(sample_fraction=sample_fraction)
        df = data['base_df']
        
        # Extract features
        df = extract_features(df, data)
        
        # Clear memory
        del data
        gc.collect()
        
        # Create target variable
        df['MORTALITY'] = df['HOSPITAL_EXPIRE_FLAG']
        
        # Select features
        exclude_cols = [
            'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
            'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOB',
            'HOSPITAL_EXPIRE_FLAG', 'MORTALITY'
        ]
        
        features = [col for col in df.columns if col not in exclude_cols]
        
        # Prepare data
        X = df[features].copy()
        y = df['MORTALITY']
        
        # Convert all features to float32
        X = X.astype('float32')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print_memory_usage()
        return X_train, X_test, y_train, y_test, features
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Set pandas options
    pd.set_option('display.max_columns', None)
    pd.options.mode.chained_assignment = None
    
    # Set environment variables
    os.environ['MALLOC_ARENA_MAX'] = '2'
    
    # Run preprocessing with 10% of data
    X_train, X_test, y_train, y_test, features = preprocess_data(sample_fraction=0.1)
    
    logger.info(f"Final shapes: Train {X_train.shape}, Test {X_test.shape}")
    logger.info(f"Number of features: {len(features)}")
    logger.info("Features: " + ", ".join(features))