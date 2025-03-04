import cv2
import heapq
import numpy as np

def find_shortest_path(image_path, start, end, clearance=5):
    """Finds the shortest path using A* algorithm and ensures a clearance from walls."""
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
    
    def neighbors(node):
        x, y = node
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        result = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if np.all(gray_image[max(0, ny-clearance):min(height, ny+clearance), 
                                     max(0, nx-clearance):min(width, nx+clearance)] > 150):
                    result.append((nx, ny))
        return result
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan Distance
    
    pq = [(0 + heuristic(start, end), 0, start)]  # (f-score, cost, node)
    distances = {start: 0}
    previous = {}
    visited_nodes = set()
    
    while pq:
        _, current_dist, current_node = heapq.heappop(pq)
        if current_node == end:
            break
        if current_node in visited_nodes:
            continue
        visited_nodes.add(current_node)
        
        for neighbor in neighbors(current_node):
            new_cost = current_dist + 1
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                f_score = new_cost + heuristic(neighbor, end)
                heapq.heappush(pq, (f_score, new_cost, neighbor))
                previous[neighbor] = current_node
    
    path = []
    node = end
    while node in previous:
        path.append(node)
        node = previous[node]
    path.reverse()
    
    if len(path) > 1:
        for i in range(len(path) - 1):
            cv2.line(image, path[i], path[i + 1], (255, 0, 0), 2)  # Blue color (BGR format)
    
    cv2.imwrite("mapped_environment_with_path.png", image)
    cv2.imshow("Path Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return path
