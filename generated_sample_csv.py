"""
Generate Sample CSV Files for Testing Dashboard
Run this to create test data you can upload to the dashboard
"""

import pandas as pd
import numpy as np

# The 41 features the model expects (before feature selection)
ALL_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

def generate_normal_traffic(n_samples=10):
    """Generate normal network traffic"""
    data = []
    for _ in range(n_samples):
        sample = {
            'duration': np.random.randint(0, 1000),
            'protocol_type': np.random.choice(['tcp', 'udp']),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'domain']),
            'flag': 'SF',
            'src_bytes': np.random.randint(100, 5000),
            'dst_bytes': np.random.randint(100, 5000),
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': 0,
            'num_failed_logins': 0,
            'logged_in': 1,
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': np.random.randint(0, 2),
            'num_shells': 0,
            'num_access_files': np.random.randint(0, 3),
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': np.random.randint(1, 50),
            'srv_count': np.random.randint(1, 50),
            'serror_rate': 0.0,
            'srv_serror_rate': 0.0,
            'rerror_rate': 0.0,
            'srv_rerror_rate': 0.0,
            'same_srv_rate': np.random.uniform(0.8, 1.0),
            'diff_srv_rate': np.random.uniform(0.0, 0.2),
            'srv_diff_host_rate': np.random.uniform(0.0, 0.1),
            'dst_host_count': np.random.randint(100, 255),
            'dst_host_srv_count': np.random.randint(100, 255),
            'dst_host_same_srv_rate': np.random.uniform(0.8, 1.0),
            'dst_host_diff_srv_rate': np.random.uniform(0.0, 0.2),
            'dst_host_same_src_port_rate': np.random.uniform(0.5, 1.0),
            'dst_host_srv_diff_host_rate': np.random.uniform(0.0, 0.1),
            'dst_host_serror_rate': 0.0,
            'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 0.0,
            'dst_host_srv_rerror_rate': 0.0
        }
        data.append(sample)
    return pd.DataFrame(data)

def generate_dos_attack(n_samples=10):
    """Generate DoS attack traffic"""
    data = []
    for _ in range(n_samples):
        sample = {
            'duration': np.random.randint(0, 10),
            'protocol_type': 'icmp',
            'service': 'eco_i',
            'flag': 'SF',
            'src_bytes': np.random.randint(1000, 50000),
            'dst_bytes': 0,
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': 0,
            'num_failed_logins': 0,
            'logged_in': 0,
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': np.random.randint(100, 511),
            'srv_count': np.random.randint(100, 511),
            'serror_rate': 0.0,
            'srv_serror_rate': 0.0,
            'rerror_rate': 0.0,
            'srv_rerror_rate': 0.0,
            'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0,
            'srv_diff_host_rate': 0.0,
            'dst_host_count': 255,
            'dst_host_srv_count': 255,
            'dst_host_same_srv_rate': 1.0,
            'dst_host_diff_srv_rate': 0.0,
            'dst_host_same_src_port_rate': 1.0,
            'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 0.0,
            'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 0.0,
            'dst_host_srv_rerror_rate': 0.0
        }
        data.append(sample)
    return pd.DataFrame(data)

def generate_probe_attack(n_samples=10):
    """Generate Probe/Scan attack traffic"""
    data = []
    for _ in range(n_samples):
        sample = {
            'duration': np.random.randint(0, 5),
            'protocol_type': 'tcp',
            'service': 'private',
            'flag': 'REJ',
            'src_bytes': 0,
            'dst_bytes': 0,
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': 0,
            'num_failed_logins': 0,
            'logged_in': 0,
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': np.random.randint(200, 400),
            'srv_count': np.random.randint(200, 400),
            'serror_rate': 1.0,
            'srv_serror_rate': 1.0,
            'rerror_rate': 0.0,
            'srv_rerror_rate': 0.0,
            'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0,
            'srv_diff_host_rate': 0.0,
            'dst_host_count': 255,
            'dst_host_srv_count': 255,
            'dst_host_same_srv_rate': 0.5,
            'dst_host_diff_srv_rate': 0.5,
            'dst_host_same_src_port_rate': 0.0,
            'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 1.0,
            'dst_host_srv_serror_rate': 1.0,
            'dst_host_rerror_rate': 0.0,
            'dst_host_srv_rerror_rate': 0.0
        }
        data.append(sample)
    return pd.DataFrame(data)

def generate_mixed_traffic(n_samples=30):
    """Generate mixed normal and attack traffic"""
    normal = generate_normal_traffic(n_samples // 2)
    dos = generate_dos_attack(n_samples // 4)
    probe = generate_probe_attack(n_samples // 4)
    
    mixed = pd.concat([normal, dos, probe], ignore_index=True)
    # Shuffle rows
    mixed = mixed.sample(frac=1).reset_index(drop=True)
    return mixed

# Generate sample files
print("Generating sample CSV files...\n")

# 1. Normal traffic only
normal_df = generate_normal_traffic(20)
normal_df.to_csv('sample_normal_traffic.csv', index=False)
print("✓ Created: sample_normal_traffic.csv (20 normal connections)")

# 2. DoS attack traffic
dos_df = generate_dos_attack(20)
dos_df.to_csv('sample_dos_attack.csv', index=False)
print("✓ Created: sample_dos_attack.csv (20 DoS attacks)")

# 3. Probe attack traffic
probe_df = generate_probe_attack(20)
probe_df.to_csv('sample_probe_attack.csv', index=False)
print("✓ Created: sample_probe_attack.csv (20 probe/scan attacks)")

# 4. Mixed traffic
mixed_df = generate_mixed_traffic(50)
mixed_df.to_csv('sample_mixed_traffic.csv', index=False)
print("✓ Created: sample_mixed_traffic.csv (50 mixed connections)")

# 5. Large batch test
large_df = generate_mixed_traffic(200)
large_df.to_csv('sample_large_batch.csv', index=False)
print("✓ Created: sample_large_batch.csv (200 connections for batch testing)")

print("\n" + "="*60)
print("SAMPLE FILES CREATED!")
print("\nFeatures included in each file: 41 network traffic attributes")
print("="*60)