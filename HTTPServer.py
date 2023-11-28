from flask import Flask, request, jsonify
from sklearn.neighbors import BallTree
import requests
import math
import heapq
import json
from urllib import parse
import threading
import logging
import traceback

from multiprocessing import Process, Manager
from SQLInstance import SQLInstance
from Node import Node
import copy

# json 형식
# {
#   "from":{"lon":"x1", "lat":"y1"}, 
#   "to":{"lon":"x2", "lat":"y2"}
# }

app = Flask(__name__)

def tmap_route(data, result_list):
    url = "https://apis.openapi.sk.com/tmap/routes/pedestrian?version=1&callback=function"

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "appKey": "5zInvza21F78XEPUAhBjZ46BqeN58rZu1Vtrq1mY"
        #e8wHh2tya84M88aReEpXCa5XTQf3xgo01aZG39k5 기존
        #5zInvza21F78XEPUAhBjZ46BqeN58rZu1Vtrq1mY
    }

    param = {
        'startX': 0,
        'startY': 0, 
        'angle' : 20,  
        'speed': 30,
        'endX': 0, 
        'endY': 0,   
        'reqCoordType': 'WGS84GEO',     # 좌표유형 [1] EPSG3857 [2] WGS84GEO [3] KATECH
        'startName': '%EC%B6%9C%EB%B0%9C',  # 출발지 명칭 URL 인코딩 
        'endName': '%EB%8F%84%EC%B0%A9',    # 도착지 명칭 URL 인코딩
        'searchOption': '0',    # [0] 추천 경로, [4] 추천 + 대로 우선, [10]최단 , [30] 최단 + 계단 제외
        'resCoordType': 'WGS84GEO',
        'sort': 'index'
    }

    param['startX'] = data['from']['lon']
    param['startY'] = data['from']['lat']
    param['endX'] = data['to']['lon']
    param['endY'] = data['to']['lat']
    searchOpt = ['4', '10']

    route = []
    for opt in searchOpt:
        param["searchOption"] = opt
        res = requests.post(url, json=param, headers=headers)
        # T맵 죽임 - 확인 필요
        route.append(res.json())

    result_list[0] = route

def hueristic(from_point, to_point):
    return abs(from_point[0] - to_point[0]) + abs(from_point[1] - to_point[1])

def get_neighbors(snode, allNodes):
    allLinks = SQLInstance().getLinks()
    return [allNodes[node] for node in allLinks[snode.id] if node in allNodes]

def a_star(start_node, end_node, allNodes, isCCTV):
    allLinks = SQLInstance().getLinks()
    weights = SQLInstance().getCCTVs() if isCCTV else SQLInstance().getCrossWalks()

    open_set = []
    close_set = set()
    
    # 시작점 생성
    start_node.hueristic = hueristic(start_node.point, end_node.point)
    start_node.score = start_node.hueristic + start_node.cost

    # 시작점 open_set에 추가
    heapq.heappush(open_set, (start_node.score, start_node))
    
    while open_set:
        score, current_node = heapq.heappop(open_set)
        # 도착점에 도착 했으면 경로 반환
        if current_node.id == end_node.id:
            path, cctv_set = [], set()
            while current_node != None:
                if current_node and current_node.parent:
                    ids = allLinks[current_node.id][current_node.parent.id][1].strip().split(' ')
                    for i in ids:
                        if i != '':
                            cctv_set.add(weights[int(i)])

                path.append(current_node.point)
                current_node = current_node.parent

            path.reverse()
            return path, list(cctv_set)
        
        # close_set에 현재점 추가
        close_set.add(current_node.id)
        neighbors = get_neighbors(current_node, allNodes)
        for nb in neighbors:
            if nb.id in close_set:
                continue

            nb_node = Node(nb.id, nb.point, current_node)

            # 노드의 적절한 비용 설정 할 것, 해당 구현에서는 현재 노드와 다음 노드 사이의 거리를 계산함
            #nb_node.cost = current_node.cost + hueristic(nb_node.point, current_node.point) # 최단 거리 코스트
            nb_node.cost = current_node.cost + (50 - allLinks[current_node.id][nb.id][0]) # CCTV 코스트
            nb_node.hueristic = hueristic(nb.point, end_node.point)
            nb_node.score = nb_node.cost + nb_node.hueristic

            # 계산한 비용보다 작은 비용이 존재한다면 추가할 필요 없음
            for _, node in open_set:
                if nb_node.id == node.id and nb_node.score >= node.score:
                    break
            else:
                heapq.heappush(open_set, (nb_node.score, nb_node))

    print("경로 없음")
    return [], []

def createNode(from_node, to_node):
    # X축 정렬, X축 같으면 Y축 정렬
    lst = sorted([from_node, to_node], key=lambda point: (point[0], point[1]))
    from_node, to_node = lst[0], lst[1]
    
    # 평행선으로 복원(각도 계산 후 음수각만큼 회전)
    deg = calculate_angle(from_node, to_node)
    a, b = rotate(from_node, to_node, -deg)
    
    # 끝 점에 오프셋 설정
    offset = 0.00455
    ld = [a[0] - offset, a[1] - offset]
    lu = [a[0] - offset, a[1] + offset]
    rd = [b[0] + offset, b[1] - offset]
    ru = [b[0] + offset, b[1] + offset]
    
    # 오프셋 점 회전(바운딩 박스 생성)
    a, b = rotate(ld, rd, deg)
    c, d = rotate(lu, ru, deg)
    
    # X 좌표와 Y 좌표의 최소값과 최대값을 찾습니다.
    x_min = min(a[0], b[0], c[0], d[0])
    x_max = max(a[0], b[0], c[0], d[0])
    y_min = min(a[1], b[1], c[1], d[1])
    y_max = max(a[1], b[1], c[1], d[1])
    
    #  노드 생성
    rst = SQLInstance().getIDXwithCoord(y_min, y_max, x_min, x_max)
    node_index = [i[0] for i in rst]
    nodes = [(i[1], i[2]) for i in rst]
    
    # 전체 바운딩 박스의 크기
    full_size = abs(CCW(a, b, c))
    
    allNodes = dict() # 경로 안에 포함될 노드
    for idx, node in enumerate(nodes):
        sz = 0
        sz += abs(CCW(a, b, node)) / 2
        sz += abs(CCW(b, c, node)) / 2
        sz += abs(CCW(c, d, node)) / 2
        sz += abs(CCW(d, a, node)) / 2

        if sz <= full_size:
            allNodes[node_index[idx]] = Node(id=node_index[idx], point=node)

    return allNodes

def calculate_angle(p1, p2):
    if p1[0] == p2[0]:
        return 90
    
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    angle_radians = math.atan(slope)
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def rotate(p1, p2, degree):
    theta = math.radians(degree)
    matrix = [[float(math.cos(theta)), float(-math.sin(theta))], [float(math.sin(theta)), float(math.cos(theta))]]

    x1 = matrix[0][0] * p1[0] + matrix[0][1] * p1[1]
    y1 = matrix[1][0] * p1[0] + matrix[1][1] * p1[1]
    x2 = matrix[0][0] * p2[0] + matrix[0][1] * p2[1]
    y2 = matrix[1][0] * p2[0] + matrix[1][1] * p2[1]

    return [x1, y1], [x2, y2]

def CCW(p1, p2, p3):
    S = p1[0] * p2[1] + p2[0] * p3[1] + p3[0] * p1[1]
    S -= p1[1] * p2[0] + p2[1] * p3[0] + p3[1] * p1[0]
    
    return S

def nearest_node(from_nd, to_nd):
    dbNode = SQLInstance().getAllCoords()
    balTree = BallTree(dbNode, leaf_size=15)

    fromNode = (float(from_nd[0]), float(from_nd[1]))
    toNode = (float(to_nd[0]), float(to_nd[1]))
    _, fromIdx = balTree.query([fromNode, toNode], k=1)
    
    return dbNode[fromIdx[0][0]], dbNode[fromIdx[1][0]]

def a_route(from_node, to_node, allNodes, isCCTV, result_list):
    sqlInstance = SQLInstance()
    nodeId = sqlInstance.getNodeIdx(from_node[1], from_node[0])
    to_nodeId = sqlInstance.getNodeIdx(to_node[1], to_node[0])

    # 신호등 가중치 구현 후 True 부분 isCCTV로 변경할 것
    result_path, node_set = a_star(Node(int(nodeId[0]), from_node), Node(int(to_nodeId[0]), to_node), allNodes, True)
    param_format = {
        'features' :[
            {
                'geometry' : {
                    'coordinates' :[],
                    'type' : 'Point'
                },
                'type' : 'Feature'
            }
        ],"type": "FeatureCollection"
    }
    
    newData_format = {
        'geometry' : {
            'coordinates' :[],
            'type' : 'LineString'
        },
        'type' : 'Feature'
    }

    path_data = copy.deepcopy(param_format)
    path_data['features'][0]['geometry']['coordinates'].extend([from_node[0], from_node[1]])
    new_path_data = copy.deepcopy(newData_format)
    new_path_data['geometry']['coordinates'].extend(result_path)
    path_data['features'].append(new_path_data)

    node_data = copy.deepcopy(param_format)
    new_node_data = copy.deepcopy(newData_format)
    new_node_data['geometry']['coordinates'].extend(node_set)
    node_data['features'].append(new_node_data)
    
    saveIdx = 1 if isCCTV else 3
    result_list[saveIdx] = path_data
    result_list[saveIdx + 1] = node_data
    
# 0.0.0.0:9000/api 주소에 POST 요청
@app.route('/api', methods=['POST'])
def process():
    try:
        data = request.get_json()
        
        from_node, to_node = nearest_node((data['from']['lon'], data['from']['lat']), (data['to']['lon'], data['to']['lat']))
        allNodes = createNode(from_node, to_node)

        manager = Manager()
        shared_list = manager.list([None, None, None, None, None])
        tmap_proc = Process(target=tmap_route, args=(data, shared_list, ))
        cctv_proc = Process(target=a_route, args=(from_node, to_node, allNodes, True, shared_list, ))
        cross_proc = Process(target=a_route, args=(from_node, to_node, allNodes, False, shared_list, ))

        tmap_proc.start()
        cctv_proc.start()
        cross_proc.start()

        tmap_proc.join()
        cctv_proc.join()
        cross_proc.join()

        response = shared_list[0]
        response.extend(shared_list[1:])

        return jsonify(response)   
    except Exception as e: # 오류 발생시 error json 타입으로 응답
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    SQLInstance()
    app.run('0.0.0.0', port=9000, threaded=True, debug=False)
