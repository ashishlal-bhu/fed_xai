import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('preprocessing')

def validate_dataframe(df, required_columns, name="DataFrame"):
    """Validate dataframe structure and contents"""
    logger.info(f"Validating {name}...")
    
    # Check if all required columns exist
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {name}: {missing_cols}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    # Log basic statistics
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"Missing values in {name}:\n{df[required_columns].isnull().sum()}")
    
    return True

def load_data():
    """Load and merge MIMIC-III tables with proper data types"""
    logger.info("Loading MIMIC-III data...")
    
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Define data types for critical columns
        dtype_dict = {
            'SUBJECT_ID': 'int64',
            'HADM_ID': 'int64',
            'ICD9_CODE': 'str',  # Ensure ICD9 codes are read as strings
            'ITEMID': 'str',     # Ensure item IDs are read as strings
            'VALUENUM': 'float64'
        }
        
        # Load core tables with specified data types
        patients = pd.read_csv(os.path.join(base_dir, 'data', 'PATIENTS.csv'))
        admissions = pd.read_csv(os.path.join(base_dir, 'data', 'ADMISSIONS.csv'))
        icustays = pd.read_csv(os.path.join(base_dir, 'data', 'ICUSTAYS.csv'))
        
        # Load clinical tables with type handling
        labevents = pd.read_csv(
            os.path.join(base_dir, 'data', 'LABEVENTS.csv'),
            dtype={'ITEMID': 'str', 'VALUENUM': 'float64'}
        )
        
        chartevents = pd.read_csv(
            os.path.join(base_dir, 'data', 'CHARTEVENTS.csv'),
            dtype={'ITEMID': 'str', 'VALUENUM': 'float64'}
        )
        
        # Load ICD tables with string handling
        procedures = pd.read_csv(
            os.path.join(base_dir, 'data', 'PROCEDURES_ICD.csv'),
            dtype={'ICD9_CODE': 'str'}
        )
        
        diagnoses = pd.read_csv(
            os.path.join(base_dir, 'data', 'DIAGNOSES_ICD.csv'),
            dtype={'ICD9_CODE': 'str'}
        )
        
        prescriptions = pd.read_csv(os.path.join(base_dir, 'data', 'PRESCRIPTIONS.csv'))
        
        # Standardize column names
        for df in [patients, admissions, icustays, labevents, chartevents, 
                  prescriptions, procedures, diagnoses]:
            df.columns = df.columns.str.upper()
        
        # Merge core patient information
        df = admissions.merge(patients, on='SUBJECT_ID', how='left')
        df = df.merge(icustays, on=['SUBJECT_ID', 'HADM_ID'], how='left')
        
        # Log data shapes for debugging
        logger.info(f"Patients shape: {patients.shape}")
        logger.info(f"Admissions shape: {admissions.shape}")
        logger.info(f"Procedures shape: {procedures.shape}")
        logger.info(f"Diagnoses shape: {diagnoses.shape}")
        
        # Verify ICD9 code types
        logger.info(f"Procedures ICD9_CODE dtype: {procedures['ICD9_CODE'].dtype}")
        logger.info(f"Diagnoses ICD9_CODE dtype: {diagnoses['ICD9_CODE'].dtype}")
        
        return {
            'base_df': df,
            'labevents': labevents,
            'chartevents': chartevents,
            'prescriptions': prescriptions,
            'procedures': procedures,
            'diagnoses': diagnoses
        }
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise

def extract_lab_features(df, labevents):
    """Extract important lab values"""
    logger.info("Processing lab events...")
    
    # Important lab items to include
    lab_items = {
        '50912': 'CREATININE',
        '50971': 'POTASSIUM',
        '50983': 'SODIUM',
        '50902': 'CHLORIDE',
        '50882': 'BICARBONATE',
        '51222': 'HEMOGLOBIN',
        '51301': 'WBC',
        '51265': 'PLATELET',
        '50931': 'GLUCOSE',
        '50960': 'MAGNESIUM',
        '50893': 'CALCIUM'
    }
    
    # Filter relevant labs and get first value per admission
    lab_df = labevents[labevents['ITEMID'].isin(lab_items.keys())]
    lab_df = lab_df.pivot_table(
        index='HADM_ID',
        columns='ITEMID',
        values='VALUENUM',
        aggfunc='first'
    ).rename(columns=lab_items)
    
    return df.merge(lab_df, on='HADM_ID', how='left')

def extract_vital_signs(df, chartevents):
    """Extract vital signs"""
    logger.info("Processing vital signs...")
    
    # Important vital signs
    vitals = {
        '220045': 'HEART_RATE',
        '220050': 'SYSTOLIC_BP',
        '220051': 'DIASTOLIC_BP',
        '220179': 'TEMPERATURE',
        '220210': 'RESPIRATORY_RATE',
        '220277': 'SPO2'
    }
    
    # Filter and pivot vital signs
    vitals_df = chartevents[chartevents['ITEMID'].isin(vitals.keys())]
    vitals_df = vitals_df.pivot_table(
        index='HADM_ID',
        columns='ITEMID',
        values='VALUENUM',
        aggfunc='first'
    ).rename(columns=vitals)
    
    return df.merge(vitals_df, on='HADM_ID', how='left')

def extract_procedures(df, procedures):
    """Extract procedure counts and categories with proper string handling"""
    logger.info("Processing procedures...")
    
    try:
        # Ensure ICD9_CODE is string type
        procedures['ICD9_CODE'] = procedures['ICD9_CODE'].astype(str)
        
        # Get procedure counts per admission
        proc_counts = procedures.groupby('HADM_ID').size().reset_index(name='PROCEDURE_COUNT')
        
        # Get top procedure categories (first 2 characters of ICD9 code)
        procedures['ICD9_CATEGORY'] = procedures['ICD9_CODE'].str[:2]
        
        # Log unique categories for debugging
        logger.info(f"Number of unique procedure categories: {procedures['ICD9_CATEGORY'].nunique()}")
        
        # Get top 20 most common categories
        top_cats = procedures['ICD9_CATEGORY'].value_counts().nlargest(20).index
        
        # Create pivot table for top categories only
        category_pivot = pd.crosstab(
            procedures['HADM_ID'],
            procedures['ICD9_CATEGORY']
        ).loc[:, top_cats]
        category_pivot = category_pivot.add_prefix('PROC_CAT_')
        
        # Merge with main dataframe
        df = df.merge(proc_counts, on='HADM_ID', how='left')
        df = df.merge(category_pivot, on='HADM_ID', how='left')
        
        # Fill NaN values with 0
        proc_columns = [col for col in df.columns if col.startswith('PROC_CAT_')]
        df[proc_columns] = df[proc_columns].fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error in procedure extraction: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise

def extract_diagnoses(df, diagnoses):
    """Extract diagnosis features with proper binary initialization"""
    logger.info("Processing diagnoses...")
    
    try:
        # Calculate Charlson comorbidity components
        charlson_codes = {
            'MI': ['410', '412'],
            'CHF': ['398.91', '402.01', '402.11', '402.91', '404.01', '404.03',
                    '404.11', '404.13', '404.91', '404.93', '425.4', '425.5',
                    '425.7', '425.8', '425.9', '428'],
            'Diabetes': ['250.0', '250.1', '250.2', '250.3', '250.4', '250.5',
                        '250.6', '250.7', '250.8', '250.9']
        }
        
        # Get diagnosis counts
        diag_counts = diagnoses.groupby('HADM_ID').size().reset_index(name='DIAGNOSIS_COUNT')
        
        # Initialize binary columns with 0
        for condition in charlson_codes.keys():
            diag_counts[f'HAS_{condition}'] = 0
        
        # Calculate comorbidity indicators
        for condition, codes in charlson_codes.items():
            # Create pattern for string matching
            patterns = [f"^{code}" for code in codes]
            pattern = '|'.join(patterns)
            
            # Find matching diagnoses
            matching_admissions = diagnoses[
                diagnoses['ICD9_CODE'].str.match(pattern, na=False)
            ]['HADM_ID'].unique()
            
            # Set indicator to 1 for matching admissions
            diag_counts.loc[
                diag_counts['HADM_ID'].isin(matching_admissions),
                f'HAS_{condition}'
            ] = 1
            
            logger.info(f"Found {len(matching_admissions)} admissions with {condition}")
        
        df = df.merge(diag_counts, on='HADM_ID', how='left')
        
        # Fill any remaining NaN values with 0 (for admissions without any diagnoses)
        binary_cols = [col for col in diag_counts.columns if col.startswith('HAS_')]
        df[binary_cols] = df[binary_cols].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error in diagnoses extraction: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise
def extract_medication_features(df, prescriptions):
    """Extract medication features with proper binary initialization"""
    logger.info("Processing medications...")
    
    try:
        # Get medication counts
        med_counts = prescriptions.groupby('HADM_ID').size().reset_index(name='MEDICATION_COUNT')
        
        # Important medication categories to track
        med_categories = {
            'ANTIBIOTIC': ['CEFAZOLIN', 'VANCOMYCIN', 'CEFTRIAXONE'],
            'CARDIAC': ['METOPROLOL', 'ASPIRIN', 'LISINOPRIL'],
            'ANALGESIC': ['MORPHINE', 'ACETAMINOPHEN', 'OXYCODONE'],
            'ANTICOAGULANT': ['HEPARIN', 'WARFARIN']
        }
        
        # Initialize all medication category columns with 0
        for category in med_categories.keys():
            med_counts[f'HAS_{category}'] = 0
        
        # Create indicators for medication categories
        for category, meds in med_categories.items():
            # Find matching prescriptions
            pattern = '|'.join(meds)
            matching_admissions = prescriptions[
                prescriptions['DRUG'].str.contains(pattern, case=False, na=False)
            ]['HADM_ID'].unique()
            
            # Set indicator to 1 for matching admissions
            med_counts.loc[
                med_counts['HADM_ID'].isin(matching_admissions),
                f'HAS_{category}'
            ] = 1
            
            logger.info(f"Found {len(matching_admissions)} admissions with {category} medications")
        
        df = df.merge(med_counts, on='HADM_ID', how='left')
        
        # Fill any remaining NaN values with 0 (for admissions without any medications)
        binary_cols = [col for col in med_counts.columns if col.startswith('HAS_')]
        df[binary_cols] = df[binary_cols].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error in medication extraction: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise

def preprocess_data(task="mortality"):
    """Main preprocessing pipeline with filtered features"""
    try:
        logger.info(f"Starting enhanced preprocessing for task: {task}")
        
        # Load and process data
        data_dict = load_data()
        df = data_dict['base_df']
        
        # Process features
        df = calculate_age_and_los(df)
        df = extract_lab_features(df, data_dict['labevents'])
        df = extract_vital_signs(df, data_dict['chartevents'])
        df = extract_procedures(df, data_dict['procedures'])
        df = extract_diagnoses(df, data_dict['diagnoses'])
        df = extract_medication_features(df, data_dict['prescriptions'])
        
        # Create target variable
        df['MORTALITY'] = df['HOSPITAL_EXPIRE_FLAG']
        
        # Define columns to exclude (metadata and identifiers)
        exclude_columns = [
            'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ROW_ID', 'row_id', 'Row_id',  # Identifiers
            'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOB',  # Time columns
            'HOSPITAL_EXPIRE_FLAG', 'MORTALITY',  # Target variable
            'index', 'Index'  # Any index columns
        ]
        
        # Select features excluding metadata columns
        features = [col for col in df.columns 
                   if col not in exclude_columns 
                   and not col.lower().startswith(('row', 'index'))]
        
        logger.info("Selected features:")
        for col in features:
            logger.info(f"- {col}")
        
        # Get feature matrix
        X = df[features].copy()
        y = df['MORTALITY']
        
        # Handle categorical features
        categorical_features = X.select_dtypes(include=['object']).columns
        logger.info(f"Encoding {len(categorical_features)} categorical features")
        
        for column in categorical_features:
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))
        
        # Clean and validate features
        logger.info("Starting feature cleaning...")
        X = clean_and_validate_features(X)
        
        # Convert to float32
        X = X.astype('float32')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Verify no metadata columns remain
        remaining_metadata = [col for col in X.columns if col.lower() in [c.lower() for c in exclude_columns]]
        if remaining_metadata:
            logger.warning(f"Found metadata columns after filtering: {remaining_metadata}")
            X_train = X_train.drop(columns=remaining_metadata)
            X_test = X_test.drop(columns=remaining_metadata)
            features = [f for f in features if f not in remaining_metadata]
        
        logger.info(f"Final dataset shape: {X.shape}")
        logger.info(f"Number of features: {len(features)}")
        logger.info("Final feature list:")
        for f in features:
            logger.info(f"- {f}")
        
        return X_train, X_test, y_train, y_test, features
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise

def calculate_age_and_los(df):
     """Calculate age and length of stay with proper datetime handling"""
     logger.info("Calculating temporal features...")
    
     try:
            # Convert datetime columns with proper error handling
        for col in ['ADMITTIME', 'DISCHTIME', 'DOB']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate age at admission
        df['AGE'] = (df['ADMITTIME'].dt.year - df['DOB'].dt.year) - \
                   ((df['ADMITTIME'].dt.month < df['DOB'].dt.month) | 
                    ((df['ADMITTIME'].dt.month == df['DOB'].dt.month) & 
                     (df['ADMITTIME'].dt.day < df['DOB'].dt.day)))
        
        # Calculate length of stay in days
        df['LOS'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / (24 * 60 * 60)
        
        # Handle invalid ages
        df.loc[df['AGE'] < 0, 'AGE'] = np.nan
        df.loc[df['AGE'] > 120, 'AGE'] = np.nan
        
        # Fill missing values with median
        df['AGE'].fillna(df['AGE'].median(), inplace=True)
        df['LOS'].fillna(df['LOS'].median(), inplace=True)
        
        logger.info(f"Age statistics:\n{df['AGE'].describe()}")
        logger.info(f"LOS statistics:\n{df['LOS'].describe()}")
        
        return df
        
     except Exception as e:
        logger.error(f"Error calculating temporal features: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise
    
def clean_and_validate_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate features with special handling for binary indicators"""
    logger.info("Starting feature cleaning and validation...")
    
    try:
        # Create a copy to avoid modifying original
        X_clean = X.copy()
        
        # Identify binary indicator columns
        binary_cols = [col for col in X_clean.columns if col.startswith('HAS_')]
        numeric_cols = [col for col in X_clean.columns if col not in binary_cols]
        
        logger.info(f"Found {len(binary_cols)} binary indicators and {len(numeric_cols)} numeric features")
        
        # Handle binary indicators first
        for col in binary_cols:
            # Ensure binary values are 0 or 1
            X_clean[col] = X_clean[col].fillna(0).astype(int)
            X_clean[col] = X_clean[col].clip(0, 1)
            
            logger.info(f"Binary feature {col} value counts:\n{X_clean[col].value_counts()}")
        
        # Handle numeric features
        for col in numeric_cols:
            try:
                # Get column stats
                col_data = X_clean[col].copy()
                col_data_clean = col_data.replace([np.inf, -np.inf], np.nan)
                
                # Calculate robust statistics
                median = col_data_clean.median()
                q1 = col_data_clean.quantile(0.25)
                q3 = col_data_clean.quantile(0.75)
                iqr = q3 - q1
                
                if pd.isna(iqr) or iqr == 0:
                    logger.warning(f"Feature {col} has zero IQR, using min-max bounds")
                    lower_bound = col_data_clean.min()
                    upper_bound = col_data_clean.max()
                    
                    # If still problematic, use 0 and 1 as bounds
                    if pd.isna(lower_bound) or pd.isna(upper_bound):
                        logger.warning(f"Using default bounds [0, 1] for feature {col}")
                        lower_bound, upper_bound = 0, 1
                else:
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                
                # Replace infinities and NaNs
                X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
                X_clean[col] = X_clean[col].fillna(median if not pd.isna(median) else 0)
                
                # Clip outliers
                X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
                
                # Log cleaning results
                logger.info(f"Cleaned feature {col}:")
                logger.info(f"  - Range: [{X_clean[col].min():.2f}, {X_clean[col].max():.2f}]")
                logger.info(f"  - Mean: {X_clean[col].mean():.2f}")
                logger.info(f"  - Std: {X_clean[col].std():.2f}")
                
            except Exception as e:
                logger.error(f"Error cleaning feature {col}: {str(e)}")
                logger.error(f"Feature statistics:")
                logger.error(f"- Unique values: {X_clean[col].nunique()}")
                logger.error(f"- Value counts:\n{X_clean[col].value_counts().head()}")
                raise
        
        # Final verification
        remaining_nans = X_clean.isna().sum()
        if remaining_nans.any():
            logger.error("Features still containing NaNs:")
            for col, count in remaining_nans[remaining_nans > 0].items():
                logger.error(f"- {col}: {count} NaN values")
            raise ValueError("NaN values still present after cleaning")
        
        if np.isinf(X_clean.select_dtypes(include=np.number)).any().any():
            raise ValueError("Infinite values still present after cleaning")
            
        logger.info(f"Final cleaned shape: {X_clean.shape}")
        return X_clean
        
    except Exception as e:
        logger.error(f"Error in feature cleaning: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise