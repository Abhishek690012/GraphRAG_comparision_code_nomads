import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessor import DataPreprocessor

async def main():
    config_path = 'config/config.yaml'
    preprocessor = DataPreprocessor(config_path)
    manifest = await preprocessor.run()
    print("Processing manifest:", manifest)

if __name__ == "__main__":
    asyncio.run(main())