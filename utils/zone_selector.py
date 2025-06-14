import cv2
import numpy as np

def get_interactive_zone_polygon(initial_frame, prompt_text="Draw Zone (C to confirm, N for next polygon, Q to finish):"):
    """
    Allows the user to interactively draw one or more polygons on a given frame.
    User clicks points, presses 'C' to complete a polygon, 'N' to start a new
    polygon (for the same zone type), or 'Q' to quit the selection for this zone type.

    Args:
        initial_frame (numpy.ndarray): The frame on which the user will draw.
        prompt_text (str): Instructions to display in the console and window title.

    Returns:
        list: A list of polygons. Each polygon is a list of [x, y] points.
              Returns an empty list if no polygons are defined or user quits.
              Returns 'QUIT_ALL' string if user explicitly presses 'Q' to quit overall.
    """
    all_polygons_for_this_type = []
    current_polygon_points = []
    
    display_frame = initial_frame.copy()
    clone_frame = initial_frame.copy() 

    window_name = "ZONE SELECTION: " + prompt_text
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon_points, display_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon_points.append((x, y))

        display_frame = clone_frame.copy()
        for poly in all_polygons_for_this_type:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(display_frame, [pts], (0, 255, 0)) # Green for defined
        
        if len(current_polygon_points) > 0:
            temp_points_to_draw = current_polygon_points + [(x, y)] if len(current_polygon_points) > 0 else []
            if len(temp_points_to_draw) > 1:
                cv2.polylines(display_frame, [np.array(temp_points_to_draw)], False, (0, 165, 255), 2) # Orange for current
            for point in current_polygon_points:
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1) # Red circles for points
        
        cv2.imshow(window_name, display_frame)

    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n--- {prompt_text} ---")
    print("  Click LEFT mouse button to define polygon points.")
    print("  Press 'C' to close the current polygon and add it.")
    print("  Press 'N' to finish current polygon(s) and move to the NEXT zone type, or start a NEW polygon for this type if nothing drawn.")
    print("  Press 'Q' to QUIT the entire zone selection process and surveillance.")
    print("  Press 'R' to RESET current polygon and redraw.")
    
    cv2.imshow(window_name, display_frame) 
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if len(current_polygon_points) >= 3:
                all_polygons_for_this_type.append(current_polygon_points)
                current_polygon_points = [] 

                display_frame = clone_frame.copy()
                for poly in all_polygons_for_this_type:
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.fillPoly(display_frame, [pts], (0, 255, 0))
                cv2.imshow(window_name, display_frame)

                print(f"  Polygon {len(all_polygons_for_this_type)} defined for this type. Press 'N' to continue or 'Q' to quit.")
            else:
                print("  Current polygon needs at least 3 points to be closed. Keep drawing.")
        
        elif key == ord('n'): 
            if len(current_polygon_points) >= 3: # If current polygon is valid, close it first
                all_polygons_for_this_type.append(current_polygon_points)
                current_polygon_points = []
                print(f"  Current polygon added. Moving to next zone type selection.")
            elif len(current_polygon_points) > 0: # If unclosed points exist but less than 3
                print("  Current polygon not closed (less than 3 points). Skipping to next zone type.")
                current_polygon_points = []
            else:
                print("  No active polygon. Moving to next zone type selection.")
            cv2.destroyWindow(window_name)
            return all_polygons_for_this_type # Exit and return collected polygons

        elif key == ord('q'): # Quit all zone selection
            print("  Quitting zone selection.")
            cv2.destroyWindow(window_name)
            return 'QUIT_ALL' # Special signal to main loop to terminate

        elif key == ord('r'): # Reset current polygon
            current_polygon_points = []
            display_frame = clone_frame.copy() # Reset display to clear current drawing
            for poly in all_polygons_for_this_type: # Redraw previously saved polygons
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.fillPoly(display_frame, [pts], (0, 255, 0))
            cv2.imshow(window_name, display_frame)
            print("  Current polygon reset. Start drawing again.")

    cv2.destroyWindow(window_name)
    return all_polygons_for_this_type