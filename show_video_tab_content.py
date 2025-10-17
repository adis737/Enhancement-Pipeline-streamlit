#!/usr/bin/env python3
"""
Show Video Tab Content
=====================

Extract and display the current video processing tab content.
"""

def show_video_tab():
    """Show the video processing tab content."""
    print("Current Video Processing Tab Content:")
    print("=" * 50)
    
    with open("streamlit_app.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find the video tab section
    in_video_tab = False
    tab_content = []
    
    for i, line in enumerate(lines):
        if "with tab2:" in line:
            in_video_tab = True
            tab_content.append(f"Line {i+1}: {line.strip()}")
            continue
        
        if in_video_tab:
            if line.strip().startswith("with tab") and "tab2" not in line:
                break
            tab_content.append(f"Line {i+1}: {line.strip()}")
    
    # Display the content
    for line in tab_content[:30]:  # Show first 30 lines
        print(line)
    
    if len(tab_content) > 30:
        print(f"... and {len(tab_content) - 30} more lines")

def check_key_features():
    """Check for key video processing features."""
    print("\nKey Features Check:")
    print("=" * 30)
    
    with open("streamlit_app.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    features = [
        ("Video Processing header", "🎬 Video Processing"),
        ("OpenCV check", "Video processing requires OpenCV"),
        ("Success message", "Video processing is available!"),
        ("Max Frame Size slider", "Max Frame Size"),
        ("Target FPS slider", "Target FPS"),
        ("Video uploader", "Choose a video file"),
        ("Enhance button", "Enhance Video"),
        ("Processing tips", "Video Processing Tips")
    ]
    
    for feature_name, search_text in features:
        if search_text in content:
            print(f"✓ {feature_name}: FOUND")
        else:
            print(f"✗ {feature_name}: NOT FOUND")

if __name__ == "__main__":
    show_video_tab()
    check_key_features()

