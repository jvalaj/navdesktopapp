import React, { useEffect, useState, useRef } from 'react';
import bgVideo from '../assets/1.mov';

export default function EntranceScreen({ onComplete }) {
    const [stage, setStage] = useState(0);
    const [showComputerLine, setShowComputerLine] = useState(false);
    const [exiting, setExiting] = useState(false);
    const videoRef = useRef(null);

    useEffect(() => {
        // Attempt to play video programmatically in case autoplay attribute is blocked
        if (videoRef.current) {
            videoRef.current.play().catch(e => console.error("Autoplay failed:", e));
        }

        // Sequence of animations
        // 0: Initial black screen (fast)
        // 1: "an AI agent that can see"
        // 2: "understand"
        // 3: "interact"
        // then: "with a computer..." (after Interact has come in)
        // 4: "welcome to nav" + Get Started button

        const timers = [];

        timers.push(setTimeout(() => setStage(1), 500));   // Start fade ins
        timers.push(setTimeout(() => setStage(2), 2500)); // "Understand"
        timers.push(setTimeout(() => setStage(3), 4500)); // "Interact"
        timers.push(setTimeout(() => setShowComputerLine(true), 5600)); // "with a computer..." after Interact has animated in
        timers.push(setTimeout(() => setStage(4), 7200)); // "Welcome to Nav" last, after sentence has animated in

        return () => timers.forEach(clearTimeout);
    }, []);

    const handleStart = () => {
        setExiting(true);
        // Wait for slide up animation to finish before unmounting
        setTimeout(() => {
            onComplete();
        }, 800);
    };

    return (
        <div className={`entrance-screen ${exiting ? 'slide-up' : ''}`}>
            <div className="entrance-bg-animation">
                <video
                    ref={videoRef}
                    src={bgVideo}
                    className="bg-video"
                    autoPlay
                    loop
                    playsInline
                // muted={false} // Audio on by default as requested
                />
                <div className="bg-overlay"></div>
            </div>

            <div className="entrance-content">
                <p className={`entrance-intro ${stage >= 1 ? 'fade-in' : ''}`}>
                An AI agent that can
                </p>

                <div className={`entrance-line ${stage >= 1 ? 'fade-in' : ''}`}>

                    <div className="entrance-hero-row mt-2">
                        <span className="hero-text">See</span>
                        <svg className="hero-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                            <circle cx="12" cy="12" r="3" />
                        </svg>
                    </div>
                </div>

                <div className={`entrance-line ${stage >= 2 ? 'fade-in' : ''}`}>
                    <div className="entrance-hero-row">
                        <span className="hero-text">Understand</span>
                        <svg className="hero-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" />
                            <path d="M9 18h6" />
                            <path d="M10 22h4" />
                        </svg>
                    </div>
                </div>

                <div className={`entrance-line ${stage >= 3 ? 'fade-in' : ''}`}>
                    <div className="entrance-hero-row">
                        <span className="hero-text">Interact</span>
                        <svg className="hero-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="m3 3 7.07 16.97 2.51-7.39 7.39-2.51L3 3z" />
                        </svg>
                    </div>
                    <p className={`entrance-computer-line text-xl block ${showComputerLine ? 'fade-in' : ''}`}>with a computer—for the very first time.</p>
                </div>

                <div className={`entrance-final ${stage >= 4 ? 'fade-in-up' : ''}`}>
                    <h1>Welcome to <span className="nav-brand">Nav</span></h1>
                    <p className="entrance-tagline">Your AI that sees, thinks, and acts on your screen.</p>

                    <button className="get-started-btn" onClick={handleStart}>
                        <span>Get Started</span>
                        <svg className="cta-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M5 12h14" />
                            <path d="m12 5 7 7-7 7" />
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    );
}
