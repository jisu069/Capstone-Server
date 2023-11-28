import pymysql

class SQLInstance:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SQLInstance, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self._createLink()
            self._createCCTV()
            self.initialized = True

    def _connectSQL(self): # DB 연결
        self.conn = pymysql.connect(host='localhost', user='admin', password='Capstone', db='test_0926')
        self.conn.set_character_set('latin1')
        return self.conn.cursor()

    def _query(self, sql, var, allOrOne):
        cur = self._connectSQL()
        cur.execute(sql, var)
        self.conn.commit()
        
        result = cur.fetchall() if allOrOne else cur.fetchone()

        self._disconnectSQL(cur=cur)
        return result
        
    def _disconnectSQL(self, cur): # DB 해제
        cur.close()
        self.conn.close()

    def _createLink(self): 
        nodes = self._query("select f_node, t_node, cctv, cctv_idx from link;", None, True)

        allLinks = dict()
        for n in nodes:
            if n[0] not in allLinks:
                allLinks[n[0]] = dict()
            allLinks[n[0]][n[1]] = (n[2], n[3])

        self.allLinks = allLinks

    def getLinks(self):
        return self.allLinks

    def _createCCTV(self):
        cctvs = self._query("select idx, lat, lon from cctv;", None, True)

        allCCTVs = dict()
        for n in cctvs:
            allCCTVs[int(n[0])] = (n[1], n[2])

        self.allCCTVs = allCCTVs

    def getCCTVs(self):
        return self.allCCTVs

    ### 해당 부분 구현 할 것
    def _createCrossWalk(self):
        self.crossWalk = None
        return None

    def getCrossWalks(self):
        return self.crossWalk
    ### 여기 까지 구현

    def getNodeIdx(self, lon, lat):
        return self._query("select NODE_IDX from node where (lon = %s) and (lat = %s);", (lon, lat), False)

    def getAllCoords(self):
        return self._query("select lat, lon from node;", None, True)

    def getIDXwithCoord(self, y_min, y_max, x_min, x_max):
        return self._query("select NODE_IDX, lat, lon from node where (lon >= %s and lon <= %s) and (lat >= %s and lat <= %s);", (y_min, y_max, x_min, x_max), True)
