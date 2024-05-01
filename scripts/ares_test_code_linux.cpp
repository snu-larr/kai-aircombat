#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
#include <errno.h>

#include <thread>

int scene_flag = 0; // 0 SAM / 1 1V1 / 2 2v2

void close_socket(int clientSocket)
{
    close(clientSocket);
}

void handle_thread(int clientSocket)
{
    int bytesReceived, bytesSent;
    float x = 0.0;
    std::string ac_data;

    char buffer[1024] = { 0 };
    std::string recv_data;

    while (1)
    {
        if (scene_flag == 0) {
            ac_data = "ORD|7011|8<1|1|100|AC|124.0|37.0|6000|1>" // 항공기 설정(8) : id, iff, upid, objtype, lon, lat, alt, op_mode
                
                // 지대공 설정 (8) : sam_id, iff, upid, obj_type, lon, lat, alt, op_dir
                "ORD|7013|8<5|2|-1|SAM|124.0|37.3|6000|50>" 

                // 항공기 기동(22) : time, id, lon, lat, alt, r, p, y, vn, ve, vu, vbx, vby, vbz, vc, G, remainingFuel, weight, thrust, distance to target, AA to target, RPM
                "ORD|8101|22<0|1|124.0|37.0|6000|0|0|0|0|0|0|800|0|0|0|1|2|3|4|5|6|7>";
        }
        else if (scene_flag == 1) {
            ac_data = "ORD|7011|8<1|1|100|0|124.0|37.0|6000|1>" // 항공기 설정(7) : id, iff, upid, objtype, lon, lat, alt, op_mode
                "ORD|7011|8<3|2|100|0|124.5|37.5|6000|0>"
            
                // 탑재장비(4) : id, iff, upid, obtype
                "ORD|7015|4<11|1|1|300>"  
                "ORD|7015|4<13|2|3|300>"
                "ORD|7015|4<15|1|1|300>"
                "ORD|7015|4<16|2|3|300>"

                // 무기 설정(4) : id, iff, upid, objtype
                "ORD|7016|4<21|1|1|400>"  
                "ORD|7016|4<22|2|3|400>"

                // 항공기 기동(22) : time, id, lon, lat, alt, r, p, y, vn, ve, vu, vbx, vby, vbz, vc, G, remainingFuel, weight, thrust, distance to target, AA to target, RPM
                "ORD|8101|22<0|1|124.0|37.0|6000|0|0|0|0|0|0|800|0|0|0|1|2|3|4|5|6|7>"
                "ORD|8101|22<0|3|124.0|37.00|6000|0|0|0|0|0|0|0|0|0|0|1|2|3|4|5|6|7>"

                // 미사일 기동(9) : time, muid, lon, lat, alt, r, p, y, v
                "ORD|8102|9<0|21|124.05|37.03|4000|0|0|0|1000>"
                "ORD|8102|9<0|22|124.15|37.03|4000|0|0|0|1000>"

                // 레이더 탐지(7) : time, ac_id, obj_type, rad_id, target_id, angle, dist
                "ORD|8201|7<0|11|500|11|3|1|2>"
                "ORD|8201|7<0|13|500|13|1|1|2>"
                
                // RWR(7) : time, ed-id, target_id, target_rad_id ,target_code ,angle, dist
                "ORD|8202|7<0|15|3|1|2|3|4>"
                "ORD|8202|7<0|16|1|1|2|3|4>";
        }
        else if (scene_flag == 2) {
           ac_data = "ORD|7011|8<1|1|100|1|124.0|37.0|6000|1>" // 항공기 설정(7) : id, iff, upid, objtype, lon, lat, alt, op_mode
                "ORD|7011|8<2|1|100|1|124.5|37.0|6000|1>"
                "ORD|7011|8<3|2|100|1|124.5|37.5|6000|0>"
                "ORD|7011|8<4|2|100|1|124.0|37.5|6000|0>"

                // 지대공 설정 (8) : sam_id, iff, upid, obj_type, lon, lat, alt, op_dir
                "ORD|7013|8<5|2|-1|200|124.0|37.5|6000|SAM>"
                "ORD|7013|8<6|2|-1|200|124.3|37.5|6000|SAM>"

                // 탑재장비(4) : id, iff, upid, obtype
                "ORD|7015|4<11|1|1|300>" 
                "ORD|7015|4<12|1|2|300>"
                "ORD|7015|4<13|2|3|300>"
                "ORD|7015|4<14|2|4|300>"
                "ORD|7015|4<15|1|1|300>"
                "ORD|7015|4<16|2|3|300>"

                // 무기 설정(4) : id, iff, upid, objtype
                "ORD|7016|4<21|1|1|400>" 
                "ORD|7016|4<22|2|3|400>"
                
                // 항공기 기동(22) : time, id, lon, lat, alt, r, p, y, vn, ve, vu, vbx, vby, vbz, vc, G, remainingFuel, weight, thrust, distance to target, AA to target, RPM
                "ORD|8101|22<0|1|124.0|37.0|6000|0|0|0|0|0|0|800|0|0|0|1|2|3|4|5|6|7>"
                "ORD|8101|22<0|2|124.0|37.05|6000|0|0|0|0|0|0|0|0|0|0|1|2|3|4|5|6|7>"
                "ORD|8101|22<0|3|124.0|37.00|6000|0|0|0|0|0|0|0|0|0|0|1|2|3|4|5|6|7>"
                "ORD|8101|22<0|4|124.0|37.05|6000|0|0|0|0|0|0|0|0|0|0|1|2|3|4|5|6|7>"

                // 미사일 기동(9) : time, muid, lon, lat, alt, r, p, y, v
                "ORD|8102|9<0|21|124.05|37.03|8000|0|0|0|1000>" 
                "ORD|8102|9<0|22|124.15|37.03|8000|0|0|0|1000>"

                // 레이더 탐지(7) : time, ac_id, obj_type, rad_id, target_id, angle, dist
                "ORD|8201|7<0|1|500|11|3|1|2>"
                "ORD|8201|7<0|2|500|13|1|1|2>"
                "ORD|8201|7<0|3|500|12|4|1|2>"
                "ORD|8201|7<0|4|500|14|2|1|2>"

                // RWR(7) : time, ed-id, target_id, target_rad_id ,target_code ,angle, dist
                "ORD|8202|7<0|15|3|1|2|3|4>"
                "ORD|8202|7<0|16|1|1|2|3|4>";
        }
        
        x += 0.001;

        ///////////////////////////////////////////
        // 첫 번째 소켓에 대한 송수신
        // Receive a number

        bytesReceived = recv(clientSocket, (char*)&buffer, sizeof(buffer), 0);

        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0'; // Null-terminate the received data
            recv_data = std::string(buffer); // Prepare next string
            std::cout << "Received: " << recv_data << std::endl;
        }
        else if (bytesReceived == 0) {
            std::cout << "Connection closing..." << std::endl;
        }
        else {
            std::cerr << "recv failed: " << errno << std::endl;
            close_socket(clientSocket);
        }

        // 받은 recv data 처리
        if (recv_data.find("ORD|9400") != std::string::npos) {
            x = 0;
            printf("RESET!!!!!");
        }

        // Send the number
        bytesSent = send(clientSocket, ac_data.c_str(), ac_data.length(), 0);
        if (bytesSent < 0) {
            std::cerr << "send failed: " << errno << std::endl;
            close_socket(clientSocket);
        }
        // std::cout << "Send: " << ac_data << std::endl;
    }
}


int main() {
    std::cout << "Ready for Communication" << std::endl;

    // 첫 번째 소켓 설정
    int listenSocket1 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in serverAddr1;
    serverAddr1.sin_family = AF_INET;
    inet_pton(AF_INET, "127.0.0.1", &serverAddr1.sin_addr);
    serverAddr1.sin_port = htons(4001);
    bind(listenSocket1, (sockaddr*)&serverAddr1, sizeof(serverAddr1));
    listen(listenSocket1, 1);

    // 두 번째 소켓 설정
    /*SOCKET listenSocket2 = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in serverAddr2;
    serverAddr2.sin_family = AF_INET;
    inet_pton(AF_INET, "127.0.0.1", &serverAddr2.sin_addr);
    serverAddr2.sin_port = htons(54001);
    bind(listenSocket2, (sockaddr*)&serverAddr2, sizeof(serverAddr2));
    listen(listenSocket2, 1);*/

    // 첫 번째 소켓 연결 수락
    int clientSocket1 = accept(listenSocket1, NULL, NULL);
    if (clientSocket1 < 0) {
        std::cerr << "Accept failed" << std::endl;
        close_socket(listenSocket1);
        return 1;
    }

    // 두 번째 소켓 연결 수락
    /*SOCKET clientSocket2 = accept(listenSocket2, NULL, NULL);
    if (clientSocket2 == INVALID_SOCKET) {
        std::cerr << "Accept failed" << std::endl;
        closesocket(listenSocket2);
        WSACleanup();
        return 1;
    }*/


    int first_send_byte;
    std::string first_data;
    
    ///
    if (scene_flag == 0) {
        first_data = "ORD|7011|8<1|1|100|AC|124.0|37.0|6000|1>" // 항공기 설정(8) : id, iff, upid, objtype, lon, lat, alt, op_mode
            "ORD|7013|8<5|2|-1|SAM|124.0|37.3|6000|50>";
    }
    else if (scene_flag == 2) {
        first_data = "ORD|7011|8<1|1|100|1|124.0|37.0|6000|1>" // 항공기 설정(8) : id, iff, upid, objtype, lon, lat, alt, op_mode
            "ORD|7011|7<2|1|1|100|1|124.5|37.0|6000|1>"
            "ORD|7011|7<3|2|100|1|124.5|37.5|6000|0>"
            "ORD|7011|7<4|2|100|1|124.0|37.5|6000|0>"

            // 지대공 설정 (8) : sam_id, iff, upid, obj_type, lon, lat, alt, op_dir
            "ORD|7013|8<5|2|-1|200|124.0|37.5|6000|SAM>"
            "ORD|7013|8<6|2|-1|200|124.3|37.5|6000|SAM>"

            // 탑재장비(4) : id, iff, upid, obtype
            "ORD|7015|4<11|1|1|300>"
            "ORD|7015|4<12|1|2|300>"
            "ORD|7015|4<13|2|3|300>"
            "ORD|7015|4<14|2|4|300>"
            "ORD|7015|4<15|1|1|300>"
            "ORD|7015|4<16|2|3|300>"

            // 무기 설정(4) : id, iff, upid, objtype
            "ORD|7016|4<21|1|1|400>"
            "ORD|7016|4<22|2|3|400>";
    }

    // Send the number
    first_send_byte = send(clientSocket1, first_data.c_str(), first_data.length(), 0);
    // std::cout << "Send: " << ac_data << std::endl;

    std::thread t1(handle_thread, clientSocket1); // 첫 번째 스레드 생성 및 실행
    // std::thread t2(handle_thread, clientSocket2); // 두 번째 스레드 생성 및 실행

    t1.join(); // 첫 번째 스레드가 완료될 때까지 대기
    // t2.join(); // 두 번째 스레드가 완료될 때까지 대기

    return 0;
}
