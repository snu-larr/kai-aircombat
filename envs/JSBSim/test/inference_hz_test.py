import random
import time

def your_function():
    # 이곳에 실제 실행하려는 함수의 로직을 넣습니다.
    
    random_value = random.random() * 0.1  # 0 ~ 1
    time.sleep(random_value)  # 함수 실행에 대략 0.01초 걸린다고 가정합니다.
    
def run_at_frequency(frequency, duration):
    interval = 1.0 / frequency  # 실행 간격
    count = 0  # 실행 횟수 카운트
    start_time = time.perf_counter()

    while time.perf_counter() - start_time < duration:
        expected_time = start_time + (count + 1) * interval
        your_function()
        count += 1
        end_time = time.perf_counter()
        sleep_time = expected_time - end_time
        print("expected time : " + str(expected_time) + " / end time : " + str(end_time))

        # print("check sleep_time : " + str(sleep_time))

        if sleep_time > 0:
            # print(sleep_time)
            # print(sleep_time)
            time.sleep(sleep_time)
        else:
            print("sleep time : " + str(sleep_time))

    return count

# 1초 동안 함수 실행하고 결과 검사
frequency = 60
duration = 10  # 1초 동안 실행
executions = run_at_frequency(frequency, duration)

print(f"Expected Executions: {frequency}")
print(f"Actual Executions: {executions}")

# 실행 횟수가 예상과 다른 경우 메시지 출력
if executions == frequency * duration:
    print("Success: The function executed 60 times in 1 second.")
else:
    print("Failure: The function did not execute 60 times in 1 second.")