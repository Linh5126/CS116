#!/usr/bin/env python3

def test_checkpoint_logic():
    """Test logic spawn food để debug vấn đề"""
    
    food_sequence = [
        (340, 530),   # Checkpoint 1 - Start area
        (340, 400),   # Checkpoint 2 - Move up  
        (704, 320),   # Checkpoint 3 - Enter danger zone
        (896, 320),   # Checkpoint 4 - Navigate through enemies
        (914, 200),   # Checkpoint 5 - Safe zone
        (1106, 200),  # Checkpoint 6 - Goal area
    ]
    
    print("=== TESTING CHECKPOINT LOGIC ===")
    print(f"Total food sequence: {len(food_sequence)} items")
    print(f"Food sequence indices: 0 to {len(food_sequence)-1}")
    print()
    
    # Simulate game flow
    for step in range(8):  # Test more than needed
        score = step
        
        print(f"--- Step {step} ---")
        print(f"Current score: {score}")
        
        # Test spawn logic
        if score < 6:
            food_pos = food_sequence[score]
            print(f"Spawn food at index {score}: {food_pos}")
        else:
            food_pos = food_sequence[-1]
            print(f"Score >= 6, use last food: {food_pos}")
        
        # Test win condition  
        if score >= 6:
            print("🎉 WIN GAME! Score >= 6")
            break
        else:
            print(f"Continue game, spawn next food...")
            
        print()

if __name__ == "__main__":
    test_checkpoint_logic() 