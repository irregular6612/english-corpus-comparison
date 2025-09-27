from typing import Any
from pathlib import Path
import json


class JsonFileHandler:
    """
    JSON 파일 처리를 담당하는 클래스
    """
    def __init__(self):
        pass
        
    def update_json_file(self, file_path: str, new_data: Any) -> None:
        """
        JSON 파일을 읽고 데이터를 추가한 후 다시 저장
        - 리스트인 경우: id가 동일한 객체가 있으면 덮어씁니다.
        - 리스트가 아닌 경우: id가 다른 객체면 리스트로 변환하여 저장합니다.
        
        Args:
            file_path (str): JSON 파일 경로
            new_data (Any): 추가할 데이터 (id 필드가 있어야 함)
        """
        try:
            # new_data가 딕셔너리이고 id 필드를 포함하는지 확인
            if not isinstance(new_data, dict) or 'id' not in new_data:
                raise ValueError("new_data는 'id' 필드를 포함하는 딕셔너리여야 합니다.")
            
            # 파일 경로 객체 생성
            file_path_obj = Path(file_path)
            
            # 디렉토리가 존재하지 않으면 생성
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # 파일이 존재하는 경우 기존 데이터 읽기
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # 데이터 추가
            if isinstance(existing_data, list):
                # id가 동일한 객체 찾기
                found = False
                for i, item in enumerate(existing_data):
                    if isinstance(item, dict) and item.get('id') == new_data['id']:
                        existing_data[i] = new_data  # 기존 데이터 덮어쓰기
                        found = True
                        break
                
                # id가 동일한 객체가 없으면 새로 추가
                if not found:
                    existing_data.append(new_data)
                    
            elif isinstance(existing_data, dict):
                # 기존 데이터가 딕셔너리이고 id가 다른 경우
                if existing_data.get('id') != new_data['id']:
                    # 기존 데이터와 새로운 데이터를 리스트로 변환
                    existing_data = [existing_data, new_data]
                else:
                    # id가 같은 경우 덮어쓰기
                    existing_data = new_data
            else:
                # 기존 데이터가 리스트나 딕셔너리가 아닌 경우
                # 새로운 데이터를 리스트로 변환하여 저장
                existing_data = [new_data]
            
            # 데이터 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
        except json.JSONDecodeError:
            print(f"JSON 파일 형식이 올바르지 않습니다: {file_path}")
        except Exception as e:
            print(f"오류 발생: {str(e)}")
    
    def load_data(self, file_path: str) -> Any:
        """
        JSON 파일에서 데이터 로드
        
        Returns:
            Any: 로드된 데이터
        """
        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {str(e)}")
            return None