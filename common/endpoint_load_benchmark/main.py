import argparse
from endpoint_load_benchmark.benchmark import LoadTester

def main():
    parser = argparse.ArgumentParser(description="HTTP Load Testing and Benchmarking Tool")
    parser.add_argument('url', type=str, help='The HTTP address to test')
    parser.add_argument('--qps', type=int, default=1, help='Queries per second')
    parser.add_argument('--model_id', type=str, help='The model ID to use for the test')
    
    args = parser.parse_args()
    
    tester = LoadTester(args.url, args.qps, args.model_id)
    tester.run()

if __name__ == '__main__':
    main()
