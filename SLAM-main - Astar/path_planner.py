import cv2
import heapq
import numpy as np
import time

def find_shortest_path(image_path, start, end):
    """Finds the shortest path using A* search algorithm, animates path drawing, and moves robot step by step."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load in color to draw the path
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape

    cv2.circle(gray_image, start, 10, (255, 255, 255), -1) 

    # Ensure start and end are in free space
    if gray_image[start[1], start[0]] <= 150:
        print("Start position is in an obstacle! Aborting pathfinding.")
        return []
    if gray_image[end[1], end[0]] <= 150:
        print("End position is in an obstacle! Choose a different destination.")
        return []

    # Mark start and end points
    cv2.circle(image, start, 5, (0, 255, 0), -1)  # Green for start
    cv2.circle(image, end, 5, (0, 0, 255), -1)  # Red for end

    def heuristic(a, b):
        """Calculates Manhattan distance as heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(node):
        x, y = node
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        result = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and gray_image[ny, nx] > 150:
                result.append((nx, ny))
        return result

    open_set = [(0 + heuristic(start, end), 0, start)]
    distances = {start: 0}
    previous = {}
    visited_nodes = set()

    while open_set:
        _, current_dist, current_node = heapq.heappop(open_set)

        if current_node in visited_nodes:
            continue

        visited_nodes.add(current_node)
        if current_node == end:
            break

        for neighbor in neighbors(current_node):
            new_cost = current_dist + heuristic(current_node, neighbor)
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                heapq.heappush(open_set, (new_cost + heuristic(neighbor, end), new_cost, neighbor))
                previous[neighbor] = current_node

    path = []
    node = end
    while node in previous:
        path.append(node)
        node = previous[node]
    path.reverse()

    # Animate path drawing
    def animate_path_drawing(image, path):
        for i in range(len(path) - 1):
            cv2.line(image, path[i], path[i + 1], (255, 0, 0), 2)  # Blue path
            cv2.imshow("Path Animation", image)
            cv2.waitKey(1)  # Animation delay

        cv2.imwrite("mapped_environment_with_path.png", image)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    # Animate robot following the path
    def follow_path(image, path):
        time.sleep(1)  # Pause before the robot starts moving
        for i, point in enumerate(path):
            image_copy = image.copy()  # Create a fresh copy each time to remove the previous green dot
            cv2.circle(image_copy, point, 5, (0, 255, 0), -1)  # Green dot at the current position
            cv2.imshow("Robot Moving", image_copy)
            cv2.waitKey(40)  # Delay between movements

    if path:
        animate_path_drawing(image, path)
        follow_path(image, path)

    return path
