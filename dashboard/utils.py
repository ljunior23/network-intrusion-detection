import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def validate_traffic_data(df):
    """
    Validate input traffic data format
    
    Args:
        df: pandas DataFrame with network traffic data
        
    Returns:
        bool: True if valid, False otherwise
        str: Error message if invalid
    """
    required_cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    return True, "Valid"


def preprocess_batch_data(df, scaler, encoders, feature_cols):
    """
    Preprocess batch data for prediction
    
    Args:
        df: pandas DataFrame with traffic data
        scaler: fitted StandardScaler
        encoders: dict of fitted LabelEncoders
        feature_cols: list of expected feature column names (34 after selection)
        
    Returns:
        numpy array: scaled feature matrix (41 features - what models actually expect)
    """
    df_processed = df.copy()
    
    # Get ALL 41 features the scaler expects (before feature selection)
    scaler_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else feature_cols
    
    # Add ALL 41 features that scaler/models expect
    for col in scaler_features:
        if col not in df_processed.columns:
            # Add missing column with appropriate default
            if col in ['protocol_type', 'service', 'flag']:
                df_processed[col] = 'tcp'  # Default categorical
            else:
                df_processed[col] = 0  # Default numeric
    
    # Reorder to match scaler's expected features (41 features)
    df_processed = df_processed[scaler_features]
    
    # Encode categorical features
    for col, encoder in encoders.items():
        if col in df_processed.columns:
            try:
                df_processed[col] = encoder.transform(df_processed[col].astype(str))
            except:
                # Handle unknown categories - use most common class (0)
                df_processed[col] = 0
    
    # Handle missing/infinite values
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with 0
    df_processed = df_processed.fillna(0)
    
    # Convert to numeric (in case any strings remain)
    df_processed = df_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Scale with all 41 features - this is what models expect!
    df_scaled = scaler.transform(df_processed)
    
    # Return ALL 41 features (models were trained on 41, not 34)
    return df_scaled


def get_severity_level(probability):
    """
    Determine severity level based on threat probability
    
    Args:
        probability: float between 0 and 1
        
    Returns:
        str: 'Low', 'Medium', or 'High'
    """
    if probability > 0.7:
        return 'High'
    elif probability > 0.4:
        return 'Medium'
    else:
        return 'Low'


def get_severity_color(severity):
    """
    Get color code for severity level
    
    Args:
        severity: str ('Low', 'Medium', 'High')
        
    Returns:
        str: color code
    """
    colors = {
        'Low': '#28a745',
        'Medium': '#ffc107',
        'High': '#dc3545'
    }
    return colors.get(severity, '#6c757d')


def format_traffic_sample(data_dict):
    """
    Format traffic data for display
    
    Args:
        data_dict: dictionary of traffic parameters
        
    Returns:
        str: formatted string
    """
    formatted = []
    for key, value in data_dict.items():
        formatted.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(formatted)


def generate_sample_traffic(attack_type='normal'):
    """
    Generate sample network traffic data with ALL 41 original features
    (Scaler needs all 41, then we select 34 for the model)
    
    Args:
        attack_type: str ('normal', 'dos', 'probe', 'r2l', 'u2r')
        
    Returns:
        dict: sample traffic parameters with all 41 original features
    """
    
    # Base template with ALL 41 original NSL-KDD features (before feature selection)
    base_sample = {
        'duration': 0,
        'protocol_type': 'tcp',
        'service': 'http',
        'flag': 'SF',
        'src_bytes': 0,
        'dst_bytes': 0,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 1,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,  # Removed by feature selection but scaler needs it
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,  # Removed by feature selection but scaler needs it
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 1,
        'srv_count': 1,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,  # Removed by feature selection but scaler needs it
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,  # Removed by feature selection but scaler needs it
        'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 255,
        'dst_host_srv_count': 255,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 1.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,  # Removed by feature selection but scaler needs it
        'dst_host_srv_serror_rate': 0.0,  # Removed by feature selection but scaler needs it
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0  # Removed by feature selection but scaler needs it
    }
    
    if attack_type == 'dos':
        base_sample.update({
            'duration': np.random.randint(0, 10),
            'protocol_type': 'icmp',
            'service': 'eco_i',
            'flag': 'SF',
            'src_bytes': np.random.randint(1000, 50000),
            'dst_bytes': 0,
            'count': np.random.randint(100, 511),
            'srv_count': np.random.randint(100, 511),
            'serror_rate': 0.0,
            'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0,
            'dst_host_count': 255,
            'dst_host_srv_count': 255,
            'dst_host_same_srv_rate': 1.0
        })
    elif attack_type == 'probe':
        base_sample.update({
            'duration': np.random.randint(0, 5),
            'protocol_type': 'tcp',
            'service': 'private',
            'flag': 'REJ',
            'src_bytes': 0,
            'dst_bytes': 0,
            'count': np.random.randint(200, 400),
            'srv_count': np.random.randint(200, 400),
            'serror_rate': 1.0,
            'srv_serror_rate': 1.0,
            'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0,
            'dst_host_count': 255,
            'dst_host_srv_count': 255,
            'dst_host_same_srv_rate': 0.5
        })
    elif attack_type == 'r2l':
        base_sample.update({
            'duration': np.random.randint(10, 100),
            'protocol_type': 'tcp',
            'service': np.random.choice(['ftp', 'telnet', 'smtp']),
            'flag': 'SF',
            'src_bytes': np.random.randint(100, 1000),
            'dst_bytes': np.random.randint(100, 1000),
            'num_failed_logins': np.random.randint(1, 5),
            'logged_in': 0,
            'count': np.random.randint(1, 10),
            'srv_count': np.random.randint(1, 10)
        })
    elif attack_type == 'u2r':
        base_sample.update({
            'duration': np.random.randint(50, 200),
            'protocol_type': 'tcp',
            'service': np.random.choice(['telnet', 'ftp', 'shell']),
            'flag': 'SF',
            'src_bytes': np.random.randint(500, 2000),
            'dst_bytes': np.random.randint(500, 2000),
            'logged_in': 1,
            'root_shell': 1,
            'su_attempted': 1,
            'num_root': np.random.randint(1, 5),
            'num_file_creations': np.random.randint(1, 3)
        })
    else:  # normal
        base_sample.update({
            'duration': np.random.randint(0, 1000),
            'protocol_type': np.random.choice(['tcp', 'udp']),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'domain', 'private']),
            'flag': 'SF',
            'src_bytes': np.random.randint(100, 5000),
            'dst_bytes': np.random.randint(100, 5000),
            'logged_in': 1,
            'count': np.random.randint(1, 50),
            'srv_count': np.random.randint(1, 50),
            'serror_rate': 0.0,
            'same_srv_rate': np.random.uniform(0.5, 1.0),
            'diff_srv_rate': np.random.uniform(0.0, 0.3),
            'dst_host_srv_count': np.random.randint(1, 255),
            'dst_host_same_srv_rate': np.random.uniform(0.8, 1.0)
        })
    
    return base_sample


def create_report(predictions_df, output_path='outputs/reports/threat_report.txt'):
    """
    Create a text report from predictions
    
    Args:
        predictions_df: DataFrame with prediction results
        output_path: str, path to save report
        
    Returns:
        str: report content
    """
    report = []
    report.append("="*80)
    report.append("NETWORK INTRUSION DETECTION REPORT")
    report.append("="*80)
    report.append(f"\nTotal Records Analyzed: {len(predictions_df)}")
    
    threats = len(predictions_df[predictions_df['Threat'] == 'Yes'])
    report.append(f"Threats Detected: {threats} ({threats/len(predictions_df)*100:.1f}%)")
    
    high = len(predictions_df[predictions_df['Severity'] == 'High'])
    medium = len(predictions_df[predictions_df['Severity'] == 'Medium'])
    low = len(predictions_df[predictions_df['Severity'] == 'Low'])
    
    report.append(f"\nSeverity Breakdown:")
    report.append(f"  High Severity: {high}")
    report.append(f"  Medium Severity: {medium}")
    report.append(f"  Low Severity: {low}")
    
    report.append(f"\nAverage Threat Probability: {predictions_df['Probability'].mean():.2%}")
    report.append(f"Max Threat Probability: {predictions_df['Probability'].max():.2%}")
    
    report.append("\n" + "="*80)
    report.append("TOP 10 HIGHEST RISK THREATS")
    report.append("="*80)
    
    top_threats = predictions_df.nlargest(10, 'Probability')
    for idx, row in top_threats.iterrows():
        report.append(f"\nRecord #{row['Record']}: {row['Probability']:.1%} - {row['Severity']} Severity")
    
    report.append("\n" + "="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Save to file
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        y_prob: prediction probabilities
        
    Returns:
        dict: metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob)
    }
    
    return metrics


def get_feature_importance(model, feature_names, top_n=10):
    """
    Get top N important features from tree-based model
    
    Args:
        model: trained model (RandomForest or XGBoost)
        feature_names: list of feature names
        top_n: number of top features to return
        
    Returns:
        pandas DataFrame: feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        df = df.sort_values('Importance', ascending=False).head(top_n)
        return df
    
    return None


def create_confusion_matrix_data(y_true, y_pred):
    """
    Create confusion matrix data for visualization
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        
    Returns:
        dict: confusion matrix values
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }


def get_recommendations(severity, probability):
    """
    Get security recommendations based on threat assessment
    
    Args:
        severity: str ('Low', 'Medium', 'High')
        probability: float (0-1)
        
    Returns:
        list: recommended actions
    """
    recommendations = []
    
    if severity == 'High':
        recommendations.extend([
            "🚨 IMMEDIATE ACTION REQUIRED",
            "Block source IP address immediately",
            "Isolate affected systems from network",
            "Initiate incident response protocol",
            "Alert security operations center (SOC)",
            "Preserve logs for forensic analysis",
            "Scan affected systems for malware"
        ])
    elif severity == 'Medium':
        recommendations.extend([
            "⚠️ ELEVATED RISK DETECTED",
            "Monitor traffic closely for escalation",
            "Log all activity from source",
            "Consider temporary rate limiting",
            "Notify security team for review",
            "Review firewall rules"
        ])
    else:
        recommendations.extend([
            "ℹ️ LOW RISK DETECTED",
            "Continue monitoring",
            "Log activity for audit trail",
            "No immediate action required"
        ])
    
    return recommendations


def format_bytes(bytes_value):
    """
    Format bytes to human readable format
    
    Args:
        bytes_value: int, number of bytes
        
    Returns:
        str: formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def export_to_csv(data, filename):
    """
    Export data to CSV file
    
    Args:
        data: pandas DataFrame
        filename: str, output filename
        
    Returns:
        str: path to saved file
    """
    import os
    output_dir = 'outputs/reports'
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    data.to_csv(filepath, index=False)
    
    return filepath


def load_sample_dataset(dataset_name='test'):
    """
    Load sample dataset for testing
    
    Args:
        dataset_name: str ('test' or 'train')
        
    Returns:
        pandas DataFrame
    """
    import os
    
    filepath = f'data/processed/{dataset_name}_labeled.csv'
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return None