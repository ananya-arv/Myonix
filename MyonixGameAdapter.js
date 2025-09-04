import React, { useState, useEffect } from 'react';

const MyonixGameAdapter = ({ children }) => {
    const [currentAdaptation, setCurrentAdaptation] = useState(null);
    const [gameParameters, setGameParameters] = useState({
        difficulty: 1.0,
        targetSize: 1.0,
        speed: 1.0,
        showRest: false
    });
    const [patientMetrics, setPatientMetrics] = useState({
        stress: 0.5,
        effort: 0.5,
        fatigue: 0.3,
        quality: 0.7
    });

    // Poll for ML recommendations (every 3 seconds)
    useEffect(() => {
        const pollRecommendations = () => {
            fetch('/api/game-commands')  // Or read from WebSocket
                .then(response => response.json())
                .then(data => {
                    if (data.action) {
                        applyAdaptation(data);
                    }
                })
                .catch(error => console.log('No new recommendations'));
        };

        const interval = setInterval(pollRecommendations, 3000);
        return () => clearInterval(interval);
    }, []);

    const applyAdaptation = (recommendation) => {
        const { action, parameters, metrics, message } = recommendation;
        
        // Update patient metrics
        setPatientMetrics(metrics);
        
        // Apply game parameter changes
        switch (action) {
            case 'increase_difficulty':
                setGameParameters(prev => ({
                    ...prev,
                    difficulty: prev.difficulty * (parameters.difficulty_multiplier || 1.2),
                    targetSize: prev.targetSize * (parameters.target_size_reduction || 0.9),
                    speed: prev.speed * (parameters.speed_increase || 1.1)
                }));
                break;
                
            case 'reduce_difficulty':
                setGameParameters(prev => ({
                    ...prev,
                    difficulty: prev.difficulty * (parameters.difficulty_multiplier || 0.7),
                    targetSize: prev.targetSize * (parameters.target_size_increase || 1.3),
                    showRest: parameters.add_rest_periods || false
                }));
                break;
                
            case 'rest_period':
                setGameParameters(prev => ({
                    ...prev,
                    showRest: true,
                    difficulty: prev.difficulty * 0.8
                }));
                
                // Auto-resume after rest period
                setTimeout(() => {
                    setGameParameters(prev => ({ ...prev, showRest: false }));
                }, parameters.pause_duration * 1000 || 30000);
                break;
                
            case 'increase_motivation':
                setGameParameters(prev => ({
                    ...prev,
                    difficulty: prev.difficulty * 0.9,
                    showRewards: true
                }));
                break;
        }
        
        // Show message to patient
        setCurrentAdaptation({ action, message });
        
        // Clear message after 5 seconds
        setTimeout(() => setCurrentAdaptation(null), 5000);
    };

    return (
        <div className="myonix-game-container">
            {/* Patient Metrics Dashboard */}
            <div className="metrics-dashboard">
                <div className="metric">
                    <span>Stress:</span>
                    <div className={`meter ${getStressLevel(patientMetrics.stress)}`}>
                        <div style={{ width: `${patientMetrics.stress * 100}%` }} />
                    </div>
                </div>
                
                <div className="metric">
                    <span>Effort:</span>
                    <div className="meter">
                        <div style={{ width: `${patientMetrics.effort * 100}%` }} />
                    </div>
                </div>
                
                <div className="metric">
                    <span>Fatigue:</span>
                    <div className="meter">
                        <div style={{ width: `${patientMetrics.fatigue * 100}%` }} />
                    </div>
                </div>
            </div>
            
            {/* Adaptation Message */}
            {currentAdaptation && (
                <div className="adaptation-message">
                    <p>{currentAdaptation.message}</p>
                </div>
            )}
            
            {/* Rest Period Overlay */}
            {gameParameters.showRest && (
                <div className="rest-overlay">
                    <h2>Take a Rest</h2>
                    <p>Relax and breathe deeply</p>
                    <div className="breathing-animation">•</div>
                </div>
            )}
            
            {/* Pass parameters to your games */}
            {React.cloneElement(children, { 
                gameParameters,
                patientMetrics,
                isResting: gameParameters.showRest
            })}
        </div>
    );
};

const getStressLevel = (stress) => {
    if (stress > 0.7) return 'high-stress';
    if (stress < 0.3) return 'low-stress';
    return 'normal-stress';
};

// Example game that uses the parameters
const TargetReachingGame = ({ gameParameters, patientMetrics, isResting }) => {
    if (isResting) return null;  // Hide game during rest
    
    const targetSize = 50 * gameParameters.targetSize;  // Adjust target size
    const gameSpeed = 1000 / gameParameters.speed;     // Adjust animation speed
    const difficulty = gameParameters.difficulty;      // Overall difficulty
    
    return (
        <div className="target-game">
            <div 
                className="target" 
                style={{
                    width: `${targetSize}px`,
                    height: `${targetSize}px`,
                    animationDuration: `${gameSpeed}ms`
                }}
            />
            {/* Game logic here */}
        </div>
    );
};

export default MyonixGameAdapter;