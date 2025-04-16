def linear_search(arr, value):
    for item in arr:
        if item == value:
            return True
    return False

def binary_search(arr, value):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == value:
            return mid
        elif arr[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def main():
    data = []

    while True:
        try:
            s = input("정수입력(0=종료, 검색하려면 -를 붙이세요): ").strip()
            if s == "":
                continue
            num = int(s)
        except:
            continue

        if num == 0:
            print("End")
            break

        if num < 0:
            target = abs(num)
            idx = binary_search(data, target)
            if idx != -1:
                print(f"{target}은 {idx + 1}번째 위치에 있습니다.")
            else:
                print("찾는 값이 없습니다.")
            continue

        if linear_search(data, num):
            print("중복값이 있습니다.")
            continue

        data.append(num)
        data.sort()
        print(data)
        print(f"총 {len(data)}개의 정수가 저장되었습니다.")

if __name__ == "__main__":
    main()