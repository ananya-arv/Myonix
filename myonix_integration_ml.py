"""
Myonix Multi-Modal Signal Processing & ML Classification
Integrates EDA, EMG, and IMU for comprehensive rehabilitation monitoring

Author: Myonix Team  
Purpose: Complete sensor fusion and adaptive exercise classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy import signal
from scipy.ndimage import median_filter
import joblib
import warnings
warnings.filterwarnings('ignore')

class MyonixMultiModalProcessor:
    """
    Complete multi-modal signal processing for Myonix rehabilitation system
    Processes EDA (Grove GSR), EMG (MyoWare 2.0), and IMU (MPU6050) simultaneously
    """
    
    def __init__(self, sampling_rate=10):
        """
        Initialize multi-modal processor
        
        Args:
            sampling_rate (int): Unified sampling rate for all sensors
        """
        self.sampling_rate = sampling_rate
        
        # Raw sensor data containers
        self.eda_raw = None
        self.emg_raw = None  
        self.imu_raw = None
        
        # Processed components
        self.eda_features = {}
        self.emg_features = {}
        self.imu_features = {}
        self.combined_features = {}
        
        # ML model for exercise adaptation
        self.ml_model = None
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Clinical thresholds (adjust based on your validation data)
        self.stress_thresholds = {'low': 0.3, 'high': 0.7}
        self.fatigue_thresholds = {'low': 0.4, 'high': 0.8}
        self.effort_thresholds = {'low': 0.2, 'high': 0.6}
        
    def load_sensor_data(self, eda_data=None, emg_data=None, imu_data=None, timestamps=None):
        """
        Load data from all sensors
        
        Args:
            eda_data (array): Grove GSR voltage readings
            emg_data (array): MyoWare EMG voltage readings  
            imu_data (dict): IMU data {'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'}
            timestamps (array): Unified timestamp vector
        """
        if timestamps is None:
            self.timestamps = np.arange(max(len(eda_data or []), len(emg_data or []), 
                                          len(imu_data.get('accel_x', []) if imu_data else []))) / self.sampling_rate
        else:
            self.timestamps = np.array(timestamps)
            
        if eda_data is not None:
            self.eda_raw = np.array(eda_data)
            print(f"Loaded EDA data: {len(self.eda_raw)} samples")
            
        if emg_data is not None:
            self.emg_raw = np.array(emg_data)
            print(f"Loaded EMG data: {len(self.emg_raw)} samples")
            
        if imu_data is not None:
            self.imu_raw = {
                'accel_x': np.array(imu_data.get('accel_x', [])),
                'accel_y': np.array(imu_data.get('accel_y', [])),
                'accel_z': np.array(imu_data.get('accel_z', [])),
                'gyro_x': np.array(imu_data.get('gyro_x', [])),
                'gyro_y': np.array(imu_data.get('gyro_y', [])),
                'gyro_z': np.array(imu_data.get('gyro_z', []))
            }
            print(f"Loaded IMU data: {len(self.imu_raw['accel_x'])} samples")
    
    def process_eda_signals(self):
        """Process EDA signals for stress and arousal detection"""
        if self.eda_raw is None:
            print("No EDA data loaded")
            return
            
        # Preprocess EDA signal
        eda_filtered = self._filter_signal(self.eda_raw, low_cutoff=1.0, high_cutoff=0.05)
        
        # Tonic-phasic decomposition
        window_samples = int(8 * self.sampling_rate)  # 8-second window
        tonic = median_filter(eda_filtered, size=window_samples)
        phasic = eda_filtered - tonic
        
        # SCR peak detection
        peaks, _ = signal.find_peaks(phasic, height=0.01, distance=int(1.0 * self.sampling_rate))
        
        # Extract EDA features
        self.eda_features = {
            # Tonic features (baseline stress)
            'eda_mean_scl': np.mean(tonic),
            'eda_std_scl': np.std(tonic),
            'eda_scl_slope': self._calculate_slope(tonic),
            
            # Phasic features (stress responses)  
            'eda_scr_frequency': len(peaks) / (len(eda_filtered)/self.sampling_rate) * 60,
            'eda_mean_scr_amplitude': np.mean(phasic[peaks]) if len(peaks) > 0 else 0,
            'eda_phasic_rms': np.sqrt(np.mean(phasic**2)),
            
            # Composite stress index
            'eda_stress_index': self._calculate_eda_stress_index(tonic, phasic, peaks)
        }
        
        print(f"EDA processing complete. Stress index: {self.eda_features['eda_stress_index']:.2f}")
        
    def process_emg_signals(self):
        """Process EMG signals for muscle activation and effort detection"""
        if self.emg_raw is None:
            print("No EMG data loaded")
            return
            
        # Preprocess EMG signal
        emg_filtered = self._filter_signal(self.emg_raw, low_cutoff=20, high_cutoff=500, filter_type='bandpass')
        
        # Rectify and smooth EMG
        emg_rectified = np.abs(emg_filtered)
        emg_envelope = self._filter_signal(emg_rectified, low_cutoff=5, filter_type='lowpass')
        
        # Calculate RMS in sliding windows
        window_size = int(0.25 * self.sampling_rate)  # 250ms windows
        emg_rms = self._calculate_sliding_rms(emg_envelope, window_size)
        
        # Detect muscle activations (bursts above threshold)
        activation_threshold = np.mean(emg_envelope) + 2 * np.std(emg_envelope)
        activations = emg_envelope > activation_threshold
        
        # Extract EMG features
        self.emg_features = {
            # Amplitude features
            'emg_mean_amplitude': np.mean(emg_envelope),
            'emg_max_amplitude': np.max(emg_envelope),
            'emg_rms': np.mean(emg_rms),
            
            # Activation features
            'emg_activation_rate': np.sum(activations) / len(activations) * 100,
            'emg_activation_duration': np.mean(self._get_activation_durations(activations)),
            
            # Frequency features
            'emg_mean_frequency': self._calculate_mean_frequency(emg_filtered),
            'emg_median_frequency': self._calculate_median_frequency(emg_filtered),
            
            # Fatigue indicators
            'emg_fatigue_index': self._calculate_emg_fatigue_index(emg_envelope),
            
            # Effort estimation (normalized to 0-1)
            'emg_effort_level': min(np.mean(emg_rms) / 0.5, 1.0)  # Normalize to max expected RMS
        }
        
        print(f"EMG processing complete. Effort level: {self.emg_features['emg_effort_level']:.2f}")
    
    def process_imu_signals(self):
        """Process IMU signals for movement analysis and exercise quality"""
        if self.imu_raw is None:
            print("No IMU data loaded")
            return
            
        # Calculate acceleration magnitude
        accel_mag = np.sqrt(self.imu_raw['accel_x']**2 + 
                           self.imu_raw['accel_y']**2 + 
                           self.imu_raw['accel_z']**2)
        
        # Calculate angular velocity magnitude  
        gyro_mag = np.sqrt(self.imu_raw['gyro_x']**2 + 
                          self.imu_raw['gyro_y']**2 + 
                          self.imu_raw['gyro_z']**2)
        
        # Motion smoothness (jerk analysis)
        accel_jerk = np.diff(accel_mag) * self.sampling_rate
        gyro_jerk = np.diff(gyro_mag) * self.sampling_rate
        
        # Range of motion estimation
        rom_x = np.ptp(np.cumsum(self.imu_raw['gyro_x']) / self.sampling_rate)  # Integrate to get angle
        rom_y = np.ptp(np.cumsum(self.imu_raw['gyro_y']) / self.sampling_rate)
        rom_z = np.ptp(np.cumsum(self.imu_raw['gyro_z']) / self.sampling_rate)
        
        # Extract IMU features
        self.imu_features = {
            # Movement intensity
            'imu_mean_accel': np.mean(accel_mag),
            'imu_max_accel': np.max(accel_mag),
            'imu_mean_gyro': np.mean(gyro_mag),
            'imu_max_gyro': np.max(gyro_mag),
            
            # Movement quality  
            'imu_accel_smoothness': 1 / (1 + np.std(accel_jerk)),  # Higher = smoother
            'imu_gyro_smoothness': 1 / (1 + np.std(gyro_jerk)),
            
            # Range of motion
            'imu_rom_total': rom_x + rom_y + rom_z,
            'imu_rom_primary': max(rom_x, rom_y, rom_z),  # Primary movement plane
            
            # Activity level
            'imu_activity_level': np.std(accel_mag) / np.mean(accel_mag),
            
            # Movement consistency
            'imu_movement_variability': np.std(gyro_mag) / np.mean(gyro_mag)
        }
        
        print(f"IMU processing complete. ROM: {self.imu_features['imu_rom_total']:.1f}°")
    
    def extract_combined_features(self):
        """Combine all sensor features into unified feature vector"""
        # Process all sensors
        self.process_eda_signals()
        self.process_emg_signals() 
        self.process_imu_signals()
        
        # Combine all features
        self.combined_features = {**self.eda_features, **self.emg_features, **self.imu_features}
        
        # Calculate composite indices
        self.combined_features.update({
            # Overall engagement (combines stress, effort, and activity)
            'engagement_index': self._calculate_engagement_index(),
            
            # Fatigue indicator (combines EMG fatigue and EDA arousal)
            'fatigue_index': self._calculate_fatigue_index(),
            
            # Exercise quality (combines EMG effort and IMU smoothness)
            'exercise_quality': self._calculate_exercise_quality(),
            
            # Adaptation recommendation
            'adaptation_recommendation': self._get_adaptation_recommendation()
        })
        
        return self.combined_features
    
    def _calculate_engagement_index(self):
        """Calculate overall patient engagement (0-1 scale)"""
        stress_component = 1 - abs(self.eda_features.get('eda_stress_index', 0.5) - 0.5) * 2  # Optimal at 0.5
        effort_component = self.emg_features.get('emg_effort_level', 0)
        activity_component = min(self.imu_features.get('imu_activity_level', 0) / 2, 1)
        
        return np.mean([stress_component, effort_component, activity_component])
    
    def _calculate_fatigue_index(self):
        """Calculate fatigue level (0-1 scale)"""
        emg_fatigue = self.emg_features.get('emg_fatigue_index', 0)
        eda_arousal_decline = max(0, 0.5 - self.eda_features.get('eda_stress_index', 0.5))
        
        return np.mean([emg_fatigue, eda_arousal_decline * 2])
    
    def _calculate_exercise_quality(self):
        """Calculate exercise performance quality (0-1 scale)"""
        effort_quality = min(self.emg_features.get('emg_effort_level', 0) / 0.7, 1)  # Optimal effort ~0.7
        movement_quality = (self.imu_features.get('imu_accel_smoothness', 0.5) + 
                           self.imu_features.get('imu_gyro_smoothness', 0.5)) / 2
        
        return np.mean([effort_quality, movement_quality])
    
    def _get_adaptation_recommendation(self):
        """Get exercise adaptation recommendation"""
        stress = self.eda_features.get('eda_stress_index', 0.5)
        fatigue = self.combined_features.get('fatigue_index', 0.5)
        engagement = self.combined_features.get('engagement_index', 0.5)
        
        if stress > 0.7 or fatigue > 0.8:
            return 'decrease_difficulty'  # Too hard
        elif stress < 0.3 and engagement < 0.4:
            return 'increase_difficulty'  # Too easy  
        elif fatigue > 0.6:
            return 'add_rest_period'     # Need break
        else:
            return 'maintain_current'     # Just right
    
    # ===== ML CLASSIFICATION SYSTEM =====
    
    def create_training_dataset(self, n_sessions=1000):
        """
        Generate synthetic training data for ML model
        Creates realistic multi-modal data with different exercise scenarios
        """
        print("Generating synthetic training dataset...")
        
        training_data = []
        labels = []
        
        # Define exercise scenarios
        scenarios = {
            'low_stress_good_form': {'stress': (0.1, 0.4), 'effort': (0.3, 0.6), 'quality': (0.7, 1.0)},
            'high_stress_struggling': {'stress': (0.7, 1.0), 'effort': (0.8, 1.0), 'quality': (0.2, 0.5)},
            'fatigued_declining': {'stress': (0.4, 0.7), 'effort': (0.2, 0.4), 'quality': (0.3, 0.6)},
            'optimal_performance': {'stress': (0.4, 0.6), 'effort': (0.5, 0.8), 'quality': (0.8, 1.0)},
            'disengaged_low_effort': {'stress': (0.1, 0.3), 'effort': (0.1, 0.3), 'quality': (0.4, 0.7)}
        }
        
        for scenario_name, ranges in scenarios.items():
            for _ in range(n_sessions // len(scenarios)):
                # Generate synthetic features within scenario ranges
                features = {
                    # EDA features
                    'eda_stress_index': np.random.uniform(*ranges['stress']),
                    'eda_scr_frequency': np.random.uniform(2, 15),
                    'eda_mean_scl': np.random.uniform(1.5, 3.5),
                    'eda_phasic_rms': np.random.uniform(0.01, 0.08),
                    
                    # EMG features  
                    'emg_effort_level': np.random.uniform(*ranges['effort']),
                    'emg_mean_amplitude': np.random.uniform(0.1, 1.0),
                    'emg_fatigue_index': np.random.uniform(0.1, 0.9),
                    'emg_activation_rate': np.random.uniform(10, 80),
                    
                    # IMU features
                    'imu_rom_total': np.random.uniform(30, 180),
                    'imu_accel_smoothness': np.random.uniform(*ranges['quality']),
                    'imu_activity_level': np.random.uniform(0.5, 2.5),
                    'imu_movement_variability': np.random.uniform(0.1, 0.8),
                    
                    # Derived features
                    'engagement_index': np.random.uniform(0.2, 1.0),
                    'exercise_quality': np.random.uniform(*ranges['quality'])
                }
                
                training_data.append(list(features.values()))
                labels.append(scenario_name)
        
        self.training_features = np.array(training_data)
        self.training_labels = np.array(labels)
        self.feature_names = list(features.keys())
        
        print(f"Generated {len(training_data)} training samples across {len(scenarios)} scenarios")
        return self.training_features, self.training_labels
    
    def train_classification_model(self, model_type='random_forest'):
        """
        Train ML model for exercise state classification
        
        Args:
            model_type (str): 'random_forest' or 'gradient_boosting'
        """
        if not hasattr(self, 'training_features'):
            self.create_training_dataset()
            
        # Prepare data
        X = self.training_features
        y = self.label_encoder.fit_transform(self.training_labels)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                          test_size=0.2, 
                                                          random_state=42, 
                                                          stratify=y)
        
        # Train model
        if model_type == 'random_forest':
            self.ml_model = RandomForestClassifier(n_estimators=100, 
                                                 max_depth=10, 
                                                 random_state=42)
        elif model_type == 'gradient_boosting':
            self.ml_model = GradientBoostingClassifier(n_estimators=100, 
                                                     max_depth=6, 
                                                     random_state=42)
        
        self.ml_model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.ml_model.score(X_train, y_train)
        test_score = self.ml_model.score(X_test, y_test)
        
        print(f"Model Training Complete!")
        print(f"Training Accuracy: {train_score:.3f}")
        print(f"Testing Accuracy: {test_score:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.ml_model, X_scaled, y, cv=5)
        print(f"Cross-validation Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.ml_model
    
    def predict_exercise_state(self, features=None):
        """
        Predict current exercise state from sensor data
        
        Args:
            features (dict): Feature dictionary, uses current session if None
        """
        if self.ml_model is None:
            raise ValueError("No trained model. Call train_classification_model() first.")
            
        if features is None:
            if not self.combined_features:
                features = self.extract_combined_features()
            else:
                features = self.combined_features
        
        # Convert to feature vector in correct order
        feature_vector = np.array([[features[name] for name in self.feature_names]])
        
        # Scale features
        feature_vector_scaled = self.feature_scaler.transform(feature_vector)
        
        # Make prediction
        prediction = self.ml_model.predict(feature_vector_scaled)[0]
        probabilities = self.ml_model.predict_proba(feature_vector_scaled)[0]
        
        # Convert back to label
        predicted_state = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores
        class_labels = self.label_encoder.classes_
        confidence_scores = dict(zip(class_labels, probabilities))
        
        return predicted_state, confidence_scores
    
    def get_real_time_recommendations(self):
        """
        Get real-time exercise adaptation recommendations
        Returns specific actions for your Myonix games
        """
        if not self.combined_features:
            self.extract_combined_features()
            
        predicted_state, confidence = self.predict_exercise_state()
        
        # Define recommendations for each state
        recommendations = {
            'low_stress_good_form': {
                'action': 'increase_challenge',
                'game_params': {
                    'difficulty_multiplier': 1.2,
                    'target_size_reduction': 0.9,
                    'speed_increase': 1.1
                },
                'message': "Great form! Let's increase the challenge."
            },
            
            'high_stress_struggling': {
                'action': 'reduce_difficulty',
                'game_params': {
                    'difficulty_multiplier': 0.7,
                    'target_size_increase': 1.3,
                    'add_rest_periods': True
                },
                'message': "Take it easy. Let's reduce the intensity."
            },
            
            'fatigued_declining': {
                'action': 'rest_period',
                'game_params': {
                    'pause_duration': 30,
                    'difficulty_multiplier': 0.8,
                    'show_relaxation_exercise': True
                },
                'message': "You seem tired. Let's take a short break."
            },
            
            'optimal_performance': {
                'action': 'maintain_current',
                'game_params': {
                    'difficulty_multiplier': 1.0,
                    'provide_positive_feedback': True
                },
                'message': "Perfect! Keep up this excellent effort."
            },
            
            'disengaged_low_effort': {
                'action': 'increase_motivation',
                'game_params': {
                    'add_gamification': True,
                    'show_progress_rewards': True,
                    'difficulty_multiplier': 0.9
                },
                'message': "Let's add some fun challenges to keep you engaged!"
            }
        }
        
        recommendation = recommendations.get(predicted_state, recommendations['optimal_performance'])
        
        return {
            'predicted_state': predicted_state,
            'confidence': max(confidence.values()),
            'recommendation': recommendation,
            'current_metrics': {
                'stress_level': self.eda_features.get('eda_stress_index', 0),
                'effort_level': self.emg_features.get('emg_effort_level', 0),
                'fatigue_level': self.combined_features.get('fatigue_index', 0),
                'exercise_quality': self.combined_features.get('exercise_quality', 0)
            }
        }
    
    def save_model(self, filepath='myonix_model.joblib'):
        """Save trained model and preprocessing objects"""
        model_data = {
            'ml_model': self.ml_model,
            'feature_scaler': self.feature_scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='myonix_model.joblib'):
        """Load pre-trained model"""
        model_data = joblib.load(filepath)
        self.ml_model = model_data['ml_model']
        self.feature_scaler = model_data['feature_scaler'] 
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print("Model loaded successfully")
    
    def visualize_session_analysis(self, figsize=(20, 12)):
        """Create comprehensive session analysis visualization"""
        if not self.combined_features:
            print("No features extracted. Run extract_combined_features() first.")
            return
            
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle('Myonix Multi-Modal Session Analysis', fontsize=16, fontweight='bold')
        
        # 1. EDA Signal Analysis
        if self.eda_raw is not None:
            axes[0,0].plot(self.timestamps[:len(self.eda_raw)], self.eda_raw, 'b-', linewidth=1)
            axes[0,0].set_title('EDA Signal')
            axes[0,0].set_ylabel('Conductance (μS)')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. EMG Signal Analysis  
        if self.emg_raw is not None:
            axes[0,1].plot(self.timestamps[:len(self.emg_raw)], self.emg_raw, 'r-', linewidth=1)
            axes[0,1].set_title('EMG Signal')
            axes[0,1].set_ylabel('Amplitude (V)')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. IMU Acceleration
        if self.imu_raw is not None:
            accel_mag = np.sqrt(self.imu_raw['accel_x']**2 + self.imu_raw['accel_y']**2 + self.imu_raw['accel_z']**2)
            axes[0,2].plot(self.timestamps[:len(accel_mag)], accel_mag, 'g-', linewidth=1)
            axes[0,2].set_title('IMU Acceleration Magnitude')
            axes[0,2].set_ylabel('Acceleration (g)')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Combined Stress/Effort Timeline
        stress_index = self.eda_features.get('eda_stress_index', 0.5)
        effort_level = self.emg_features.get('emg_effort_level', 0.5)
        
        timeline_x = [0, 1, 2, 3]
        timeline_stress = [stress_index] * 4
        timeline_effort = [effort_level] * 4
        
        axes[0,3].plot(timeline_x, timeline_stress, 'orange', linewidth=3, label='Stress')
        axes[0,3].plot(timeline_x, timeline_effort, 'purple', linewidth=3, label='Effort')
        axes[0,3].axhline(y=0.3, color='g', linestyle='--', alpha=0.5)
        axes[0,3].axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
        axes[0,3].set_title('Stress & Effort Levels')
        axes[0,3].legend()
        axes[0,3].grid(True, alpha=0.3)
        
        # 5-8. Feature Radar Charts
        feature_categories = {
            'Stress Indicators': ['eda_stress_index', 'eda_scr_frequency', 'eda_phasic_rms'],
            'Effort Indicators': ['emg_effort_level', 'emg_mean_amplitude', 'emg_activation_rate'],
            'Movement Quality': ['imu_accel_smoothness', 'imu_rom_total', 'exercise_quality'],
            'Fatigue Indicators': ['emg_fatigue_index', 'fatigue_index', 'engagement_index']
        }
        
        for idx, (category, features) in enumerate(feature_categories.items()):
            ax = axes[1, idx]
            values = [self.combined_features.get(feat, 0.5) for feat in features]
            
            ax.bar(range(len(features)), values, alpha=0.7, color=plt.cm.Set3(idx))
            ax.set_title(category)
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels([f.split('_')[-1] for f in features], rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        # 9. ML Prediction Results
        if self.ml_model is not None:
            try:
                predicted_state, confidence = self.predict_exercise_state()
                
                ax = axes[2,0]
                states = list(confidence.keys())
                probs = list(confidence.values())
                
                bars = ax.bar(range(len(states)), probs, alpha=0.7, color='skyblue')
                bars[np.argmax(probs)].set_color('orange')  # Highlight prediction
                
                ax.set_title(f'Exercise State Prediction\nPredicted: {predicted_state}')
                ax.set_xticks(range(len(states)))
                ax.set_xticklabels([s.replace('_', '\n') for s in states], rotation=0, ha='center')
                ax.set_ylabel('Probability')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax = axes[2,0]
                ax.text(0.5, 0.5, f'ML Model Error:\n{str(e)}', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10)
                ax.set_title('ML Prediction (Error)')
        
        # 10. Recommendations Dashboard
        try:
            recommendations = self.get_real_time_recommendations()
            
            ax = axes[2,1]
            ax.text(0.05, 0.95, f"""Current State: {recommendations['predicted_state'].replace('_', ' ').title()}
Confidence: {recommendations['confidence']:.1%}

Action: {recommendations['recommendation']['action'].replace('_', ' ').title()}

Current Metrics:
• Stress: {recommendations['current_metrics']['stress_level']:.2f}
• Effort: {recommendations['current_metrics']['effort_level']:.2f}
• Fatigue: {recommendations['current_metrics']['fatigue_level']:.2f}
• Quality: {recommendations['current_metrics']['exercise_quality']:.2f}

Message: {recommendations['recommendation']['message']}""",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax.set_title('Real-time Recommendations')
            ax.axis('off')
            
        except Exception as e:
            ax = axes[2,1]
            ax.text(0.5, 0.5, f'Recommendation Error:\n{str(e)}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=10)
            ax.set_title('Recommendations (Error)')
        
        # 11. Session Summary Statistics
        ax = axes[2,2]
        summary_text = f"""SESSION SUMMARY

Duration: {len(self.timestamps)//self.sampling_rate:.0f}s

EDA Metrics:
• Stress Index: {self.eda_features.get('eda_stress_index', 0):.2f}
• SCR Frequency: {self.eda_features.get('eda_scr_frequency', 0):.1f}/min

EMG Metrics:
• Effort Level: {self.emg_features.get('emg_effort_level', 0):.2f}
• Fatigue Index: {self.emg_features.get('emg_fatigue_index', 0):.2f}

IMU Metrics:
• Total ROM: {self.imu_features.get('imu_rom_total', 0):.1f}°
• Movement Quality: {self.imu_features.get('imu_accel_smoothness', 0):.2f}

OVERALL:
• Engagement: {self.combined_features.get('engagement_index', 0):.2f}
• Exercise Quality: {self.combined_features.get('exercise_quality', 0):.2f}"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Session Summary')
        ax.axis('off')
        
        # 12. Adaptation Timeline
        ax = axes[2,3]
        adaptation_history = ['maintain', 'increase', 'maintain', 'decrease', 'rest']
        adaptation_colors = {'maintain': 'gray', 'increase': 'green', 'decrease': 'red', 'rest': 'blue'}
        
        for i, action in enumerate(adaptation_history):
            ax.bar(i, 1, color=adaptation_colors[action], alpha=0.7, width=0.8)
        
        ax.set_title('Adaptation History')
        ax.set_xticks(range(len(adaptation_history)))
        ax.set_xticklabels([f'T{i+1}' for i in range(len(adaptation_history))])
        ax.set_ylabel('Action')
        ax.set_yticks([])
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=action.title()) 
                         for action, color in adaptation_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    # ===== UTILITY FUNCTIONS =====
    
    def _filter_signal(self, signal_data, low_cutoff=None, high_cutoff=None, filter_type='bandpass', order=4):
        """Apply Butterworth filter to signal"""
        nyquist = self.sampling_rate / 2
        
        if filter_type == 'lowpass' and low_cutoff:
            b, a = signal.butter(order, low_cutoff/nyquist, btype='low')
        elif filter_type == 'highpass' and high_cutoff:
            b, a = signal.butter(order, high_cutoff/nyquist, btype='high') 
        elif filter_type == 'bandpass' and low_cutoff and high_cutoff:
            b, a = signal.butter(order, [high_cutoff/nyquist, low_cutoff/nyquist], btype='band')
        else:
            return signal_data
            
        return signal.filtfilt(b, a, signal_data)
    
    def _calculate_slope(self, signal_data):
        """Calculate linear trend slope"""
        x = np.arange(len(signal_data))
        slope, _ = np.polyfit(x, signal_data, 1)
        return slope
    
    def _calculate_sliding_rms(self, signal_data, window_size):
        """Calculate RMS in sliding windows"""
        rms_values = []
        for i in range(len(signal_data) - window_size + 1):
            window = signal_data[i:i+window_size]
            rms_values.append(np.sqrt(np.mean(window**2)))
        return np.array(rms_values)
    
    def _get_activation_durations(self, binary_signal):
        """Get durations of continuous activation periods"""
        durations = []
        current_duration = 0
        
        for value in binary_signal:
            if value:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration / self.sampling_rate)
                    current_duration = 0
        
        return durations if durations else [0]
    
    def _calculate_mean_frequency(self, signal_data):
        """Calculate mean frequency using power spectral density"""
        freqs, psd = signal.welch(signal_data, self.sampling_rate)
        return np.sum(freqs * psd) / np.sum(psd)
    
    def _calculate_median_frequency(self, signal_data):
        """Calculate median frequency"""
        freqs, psd = signal.welch(signal_data, self.sampling_rate)
        cumulative_psd = np.cumsum(psd)
        median_idx = np.where(cumulative_psd >= cumulative_psd[-1] / 2)[0][0]
        return freqs[median_idx]
    
    def _calculate_emg_fatigue_index(self, emg_envelope):
        """Calculate EMG fatigue index based on amplitude decline"""
        # Split signal into first and last thirds
        third = len(emg_envelope) // 3
        first_third = np.mean(emg_envelope[:third])
        last_third = np.mean(emg_envelope[-third:])
        
        # Fatigue = relative decrease in amplitude
        if first_third > 0:
            fatigue_index = max(0, (first_third - last_third) / first_third)
        else:
            fatigue_index = 0
            
        return min(fatigue_index, 1.0)
    
    def _calculate_eda_stress_index(self, tonic, phasic, peaks):
        """Calculate EDA stress index (0-1 scale)"""
        # Normalize components
        scr_freq = len(peaks) / (len(phasic)/self.sampling_rate) * 60  # per minute
        scr_freq_norm = min(scr_freq / 20, 1)  # Normalize to max 20/min
        
        scr_amp = np.mean(phasic[peaks]) if len(peaks) > 0 else 0
        scr_amp_norm = min(scr_amp / 0.1, 1)  # Normalize to max 0.1
        
        phasic_rms = np.sqrt(np.mean(phasic**2))
        phasic_norm = min(phasic_rms / 0.05, 1)  # Normalize to max 0.05
        
        # Weighted combination
        stress_index = 0.4 * scr_freq_norm + 0.3 * scr_amp_norm + 0.3 * phasic_norm
        return min(stress_index, 1.0)


# ===== DEMO FUNCTION =====

def demo_myonix_system():
    """
    Complete demo of Myonix multi-modal processing and ML classification
    """
    print("=== MYONIX MULTI-MODAL DEMO ===\n")
    
    # Initialize processor
    processor = MyonixMultiModalProcessor(sampling_rate=10)
    
    # Generate synthetic multi-modal data (simulating 2-minute exercise session)
    duration = 120  # seconds
    n_samples = duration * 10  # 10 Hz sampling
    
    print("1. Generating synthetic sensor data...")
    
    # Synthetic EDA data (stress response during exercise)
    time_vec = np.arange(n_samples) / 10
    eda_baseline = 2.5 + 0.3 * np.sin(2 * np.pi * time_vec / 60)  # Slow variation
    stress_events = np.random.exponential(15, 8)  # Stress events every ~15 seconds
    stress_events = np.cumsum(stress_events)
    stress_events = stress_events[stress_events < duration] * 10  # Convert to sample indices
    
    eda_signal = eda_baseline.copy()
    for event_idx in stress_events.astype(int):
        if event_idx < len(eda_signal):
            # Add SCR response
            for i in range(min(50, len(eda_signal) - event_idx)):  # 5-second response
                eda_signal[event_idx + i] += 0.05 * np.exp(-i/20)
    
    eda_signal += np.random.normal(0, 0.01, n_samples)  # Add noise
    
    # Synthetic EMG data (muscle activation during exercises)
    emg_baseline = 0.1 + np.random.normal(0, 0.02, n_samples)
    exercise_periods = [(20, 40), (60, 80), (100, 120)]  # Three exercise sets
    
    emg_signal = emg_baseline.copy()
    for start, end in exercise_periods:
        start_idx, end_idx = start * 10, end * 10
        # Add muscle activation with fatigue
        activation_strength = np.linspace(0.8, 0.4, end_idx - start_idx)  # Decreasing due to fatigue
        emg_signal[start_idx:end_idx] += activation_strength * (0.3 + 0.2 * np.random.random(end_idx - start_idx))
    
    # Synthetic IMU data (arm movements during exercises)  
    imu_data = {
        'accel_x': np.random.normal(0, 0.5, n_samples),
        'accel_y': np.random.normal(9.8, 1.0, n_samples),  # Gravity + movements
        'accel_z': np.random.normal(0, 0.3, n_samples),
        'gyro_x': np.random.normal(0, 10, n_samples),
        'gyro_y': np.random.normal(0, 15, n_samples), 
        'gyro_z': np.random.normal(0, 5, n_samples)
    }
    
    # Add movement patterns during exercise periods
    for start, end in exercise_periods:
        start_idx, end_idx = start * 10, end * 10
        # Add rhythmic movements
        movement_freq = 0.5  # 0.5 Hz movement
        t = np.arange(end_idx - start_idx) / 10
        imu_data['accel_x'][start_idx:end_idx] += 2 * np.sin(2 * np.pi * movement_freq * t)
        imu_data['gyro_y'][start_idx:end_idx] += 30 * np.sin(2 * np.pi * movement_freq * t)
    
    # Load data into processor
    print("2. Loading multi-modal sensor data...")
    processor.load_sensor_data(eda_data=eda_signal, 
                              emg_data=emg_signal, 
                              imu_data=imu_data,
                              timestamps=time_vec)
    
    # Extract all features
    print("3. Processing all sensor signals and extracting features...")
    features = processor.extract_combined_features()
    
    print(f"\nExtracted {len(features)} features:")
    for feature, value in list(features.items())[:10]:  # Show first 10
        print(f"  {feature}: {value:.3f}")
    print("  ...")
    
    # Train ML model
    print("\n4. Training ML classification model...")
    processor.train_classification_model(model_type='random_forest')
    
    # Make predictions
    print("\n5. Making real-time predictions...")
    predicted_state, confidence = processor.predict_exercise_state()
    print(f"Predicted State: {predicted_state}")
    print(f"Confidence Scores:")
    for state, conf in confidence.items():
        print(f"  {state}: {conf:.2%}")
    
    # Get recommendations
    print("\n6. Generating adaptation recommendations...")
    recommendations = processor.get_real_time_recommendations()
    
    print(f"\nRECOMMENDATIONS:")
    print(f"Action: {recommendations['recommendation']['action']}")
    print(f"Message: {recommendations['recommendation']['message']}")
    print(f"Game Parameters: {recommendations['recommendation']['game_params']}")
    
    # Create comprehensive visualization
    print("\n7. Creating comprehensive analysis visualization...")
    processor.visualize_session_analysis()
    
    # Save model for future use
    print("\n8. Saving trained model...")
    processor.save_model('myonix_demo_model.joblib')
    
    print("\n=== DEMO COMPLETE ===")
    print("The Myonix system successfully processed multi-modal sensor data,")
    print("extracted clinically relevant features, classified exercise states,")
    print("and provided real-time adaptation recommendations!")
    
    return processor

# Uncomment to run the complete demo:
# processor = demo_myonix_system()

print("Myonix Multi-Modal System Ready!")
print("Run demo_myonix_system() to see the complete workflow in action.")
                    