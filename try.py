from multiprocessing import Process, Lock, current_process, Value
import time


def process_with_shared_resource(shared_resource):
    name = current_process().name
    try:
        print(f"{name} 获取当前共享变量： {shared_resource.value}")
        shared_resource.value += 1
        time.sleep(0.1)
        print(f"{name} 更新后的共享变量： {shared_resource.value}")
    finally:
        pass

if __name__ == "__main__":
    # 创建一个共享的资源和锁
    shared_resource = Value('i', 0)

    # 创建多个进程
    processes = []
    for _ in range(5):
        p = Process(target=process_with_shared_resource, args=(shared_resource,))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print(f"最终变量值: {shared_resource.value}")

