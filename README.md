# Capstone 
보행자용 내비게이션 (김건희(Android), 김민우(iOS), 서준형(Server), 전지수(Server))<br/>
다양한 경로를 고려하는 것보다 최단 경로 위주의 경로만을 제공, 보행자 데이터를 활용한 새로운 경로 제공<br/>
CCTV, 보안등 데이터를 이용하여 시스템에서 설정한 가중치로 경로를 계산

O 2023Capstone Server
  - 서버 구현 W. 서준형
  - 서버 OS : Debian(11)
  - 언어 : Python(3.9.2)
  - DB : MariaDB(10.6.12)

O 서버 구축 Docker 
  - https://hub.docker.com/layers/jisu069/server/0.4/images/sha256-6f5fafa7e9c06b347fa9a23184621d2d6cdb811bd336381acb3f577827fe272d?context=repo

O 서버 코드
  - HTTPServer
    * HTTP 서버, 경로 계산, 클라이언트 요청/응답
    * Astar, CCW를 이용한 환경 데이터 경로와 TMap API 경로 제공
  - Node
    * 노드 초기화 클래스
  - SQLInstance
    * DB 연결 및 사용 관련 클래스
    * 여러 SQL 질의문을 한 번에 실행하여 시간 요소를 줄이기 위해 클래스로 분리
