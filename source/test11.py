def main():
    numbers = []

    while True:
        try:
            num = int(input("정수를 입력하세요 (0 입력 시 종료): "))

            if num == 0:
                print(f"총 {len(numbers)}개의 정수가 저장되었습니다")
                break

            if num < 0:
                # 음수일 경우 해당 양수의 위치 출력
                abs_num = abs(num)
                if abs_num in numbers:
                    index = numbers.index(abs_num)
                    print(f"{abs_num}는 리스트의 {index}번째(0부터 시작) 위치에 있습니다.")
                else:
                    print("찾는 값이 없습니다.")
                continue

            # 중복값 검사 (선형탐색)
            if num in numbers:
                print("중복값이 있습니다.")
                continue

            # 오름차순 삽입 (정렬된 상태 유지)
            inserted = False
            for i in range(len(numbers)):
                if num < numbers[i]:
                    numbers.insert(i, num)
                    inserted = True
                    break
            if not inserted:
                numbers.append(num)

        except ValueError:
            print("정수를 입력해주세요.")

if __name__ == "__main__":
    main()
