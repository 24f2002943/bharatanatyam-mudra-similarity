const videoElement = document.getElementById('webcam-feed');
const canvasElement = document.getElementById('canvas-overlay');
const canvasCtx = canvasElement.getContext('2d');
const detectedMudraName = document.getElementById('detected-mudra-name');
const detectedMudraMeaning = document.getElementById('detected-mudra-meaning');
const confidenceValue = document.getElementById('confidence-value');
const statusBadge = document.getElementById('status-badge');
const loadingOverlay = document.getElementById('loading-overlay');
const voiceToggleBtn = document.getElementById('voice-toggle');

function setStatus(statusClass, message) {
    if (statusBadge) {
        statusBadge.className = `status-badge ${statusClass}`;
        statusBadge.textContent = message;
    }
}

let hands = null;
let lastCallTime = 0;
let lastDetected = '';
let voiceEnabled = true;
let indianVoice = null;

// New Logic State
let isChallengeMode = false;
let challengeTarget = '';
let currentStabilizingName = '';
let stabilizationStartTime = 0;
const STABILIZATION_MS = 1500; // 1.5 seconds

const challengeBtn = document.getElementById('challenge-btn');
const challengeBox = document.getElementById('challenge-box');
const challengeTargetUI = document.getElementById('challenge-target');
const detectionDetails = document.getElementById('detection-details');
const clarityBar = document.getElementById('clarity-bar');

if (challengeBtn) {
    challengeBtn.addEventListener('click', () => {
        isChallengeMode = !isChallengeMode;
        if (isChallengeMode) {
            challengeBtn.textContent = 'Stop Challenge';
            challengeBtn.style.background = 'var(--marigold)';
            challengeBtn.style.color = 'var(--dark-bg)';
            challengeBox.style.display = 'block';
            detectionDetails.style.display = 'none';
            pickNewChallenge();
        } else {
            challengeBtn.textContent = 'Start Challenge';
            challengeBtn.style.background = 'transparent';
            challengeBtn.style.color = 'var(--warm-ivory)';
            challengeBox.style.display = 'none';
            detectionDetails.style.display = 'block';
            challengeTarget = '';
        }
    });
}

function pickNewChallenge() {
    if (window.AVAILABLE_MUDRAS && window.AVAILABLE_MUDRAS.length > 0) {
        const idx = Math.floor(Math.random() * window.AVAILABLE_MUDRAS.length);
        challengeTarget = window.AVAILABLE_MUDRAS[idx];
        challengeTargetUI.textContent = challengeTarget;
        challengeTargetUI.style.color = 'var(--warm-ivory)';
    }
}

// Setup Voice Synthesis
function initVoice() {
    const voices = window.speechSynthesis.getVoices();
    // Look for Indian English or Hindi for correct phonetics
    indianVoice = voices.find(v => v.lang === 'hi-IN' || v.lang === 'en-IN') || voices[0];
}

if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = initVoice;
    initVoice();
} else {
    voiceToggleBtn.style.display = 'none'; // Hide if not supported
}

voiceToggleBtn.addEventListener('click', () => {
    voiceEnabled = !voiceEnabled;
    voiceToggleBtn.textContent = voiceEnabled ? '🔊 Voice On' : '🔇 Voice Off';
    voiceToggleBtn.className = voiceEnabled ? 'voice-btn active' : 'voice-btn';
    if (!voiceEnabled) window.speechSynthesis.cancel();
});

function speakMudra(text) {
    if (!voiceEnabled || !('speechSynthesis' in window)) return;
    
    // Cancel any ongoing speech so it doesn't stutter if mudra changes fast
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    if (indianVoice) utterance.voice = indianVoice;
    utterance.rate = 0.9; // Slightly slower for clarity
    utterance.pitch = 1.1;
    
    window.speechSynthesis.speak(utterance);
}

// Initialize App independently of AI load
async function initApp() {
    try {
        // 1. Start camera IMMEDIATELY so user isn't waiting
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: "user" }
        });
        videoElement.srcObject = stream;
        
        await new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                resolve();
            };
        });

        // Ensure canvas matches video size once playing
        canvasElement.width = videoElement.clientWidth;
        canvasElement.height = videoElement.clientHeight;

        // 2. Initialize MediaPipe (downloads the large model in background)
        hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });

        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.6
        });

        hands.onResults(onResults);

        // 3. Process video frames
        let isProcessing = false;
        async function processFrame() {
            if (!videoElement.paused && !videoElement.ended) {
                if (!isProcessing) {
                    isProcessing = true;
                    try {
                        // hands.send is what triggers the model download on the very first call
                        await hands.send({ image: videoElement });
                    } catch (err) {
                        console.error('Error sending frame to MediaPipe:', err);
                    }
                    isProcessing = false;
                }
            }
            requestAnimationFrame(processFrame);
        }

        // Start processing loop
        processFrame();

    } catch (err) {
        console.error("Camera access error:", err);
        setStatus('error', 'Camera Access Denied');
        if(loadingOverlay) loadingOverlay.textContent = "Camera Blocked";
    }
}

let isModelReady = false;

function normalizeLandmarks(landmarks) {
    // 1. Translate wrist to origin (landmark 0)
    const wrist = landmarks[0];
    const translated = landmarks.map(lm => ({
        x: lm.x - wrist.x,
        y: lm.y - wrist.y,
        z: lm.z - wrist.z
    }));
    
    // 2. Scale by distance from wrist to middle-finger-MCP (landmark 9)
    const middleMCP = translated[9];
    const scale = Math.sqrt(
        middleMCP.x * middleMCP.x +
        middleMCP.y * middleMCP.y +
        middleMCP.z * middleMCP.z
    );
    
    const normalized = scale > 0 
        ? translated.map(lm => ({
            x: lm.x / scale,
            y: lm.y / scale,
            z: lm.z / scale
        }))
        : translated;
        
    return normalized;
}


function onResults(results) {
    if (!isModelReady) {
        isModelReady = true;
        // Hide the "Awakening AI..." text only when the AI has successfully processed the first frame
        if (loadingOverlay) loadingOverlay.style.display = 'none';
        setStatus('waiting', 'Awaiting Gesture...');
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];

        // Draw landmarks with gold/crimson cultural colors
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#DAA520', lineWidth: 4 });
        drawLandmarks(canvasCtx, landmarks, { color: '#8B0000', lineWidth: 2, radius: 4 });

        const normalized = normalizeLandmarks(landmarks);
        const flatLandmarks = normalized.flatMap(lm => [lm.x, lm.y, lm.z]);

        // Rate-limit API calls
        const now = Date.now();
        if (now - lastCallTime > 150) {
            lastCallTime = now;
            setStatus('detecting', 'Analyzing Hastas...');

            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ landmarks: flatLandmarks })
            })
            .then(res => res.json())
            .then(data => {
                if (data.predicted) {
                    const name = data.predicted;
                    const conf = data.confidence || 0;
                    
                    // Update Clarity Meter instantly
                    confidenceValue.textContent = conf > 0 ? `${conf}%` : '—';
                    if (clarityBar) {
                        clarityBar.style.width = `${conf}%`;
                        if (conf > 85) clarityBar.style.background = '#4CAF50'; // Green
                        else if (conf > 60) clarityBar.style.background = '#FF9933'; // Marigold
                        else clarityBar.style.background = '#8B0000'; // Crimson
                    }

                    // Stabilization Logic
                    if (name !== currentStabilizingName) {
                        currentStabilizingName = name;
                        stabilizationStartTime = now;
                        detectedMudraName.style.opacity = 0.5; // Visual cue of instability
                        setStatus('detecting', `Locking: ${name}...`);
                    } else if (now - stabilizationStartTime >= STABILIZATION_MS) {
                        // Successfully held for 1.5s
                        detectedMudraName.style.opacity = 1;

                        if (isChallengeMode) {
                            if (name === challengeTarget && conf > 70) {
                                // Challenge Success!
                                challengeTargetUI.style.color = '#4CAF50';
                                challengeTargetUI.textContent = "✓ " + name;
                                speakMudra("Correct");
                                setStatus('active', 'Perfect Match!');
                                
                                // Reset holding to prevent multiple triggers
                                stabilizationStartTime = now + 1000; 
                                
                                setTimeout(() => {
                                    pickNewChallenge();
                                }, 2000);
                            }
                        } else {
                            // Normal Free Practice Mode State change logic
                            if (name !== lastDetected) {
                                detectedMudraName.classList.add('pop');
                                setTimeout(() => detectedMudraName.classList.remove('pop'), 400);
                                lastDetected = name;
                                speakMudra(name);
                            }
                            detectedMudraName.textContent = name;
                            detectedMudraMeaning.textContent = data.meaning;
                            setStatus('active', 'Recognized ✓');
                        }
                    }
                }
            })
            .catch(err => {
                console.error('Prediction error:', err);
                setStatus('error', 'Disruption in connection');
            });
        }
    } else {
        // No hand found
        currentStabilizingName = '';
        if (!isChallengeMode) {
            detectedMudraName.textContent = '—';
            detectedMudraMeaning.textContent = '—';
        }
        confidenceValue.textContent = '0%';
        if(clarityBar) {
            clarityBar.style.width = '0%';
            clarityBar.style.background = '#DAA520';
        }
        
        if (lastDetected !== '') lastDetected = '';
        
        setStatus('waiting', 'Awaiting Gesture...');
    }

    canvasCtx.restore();
}

// Resize canvas with video
window.addEventListener('resize', () => {
    canvasElement.width = videoElement.clientWidth;
    canvasElement.height = videoElement.clientHeight;
});

// Start if we are on the practice page
if (videoElement) {
    initApp();
}
