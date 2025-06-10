"""
API 응답 매퍼 기본 클래스

모든 API 응답 매퍼의 공통 기능을 제공합니다:
- 데이터 검증
- 에러 처리
- 로깅
- 캐싱
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from datetime import datetime, timedelta
import hashlib
import json
import threading

logger = logging.getLogger(__name__)

# 제네릭 타입 정의
T = TypeVar('T')  # 도메인 엔터티 타입
ApiResponse = Dict[str, Any]


class MappingError(Exception):
    """매핑 중 발생하는 예외"""
    
    def __init__(self, message: str, api_response: Optional[Dict[str, Any]] = None):
        self.api_response = api_response
        super().__init__(message)


class BaseMapper(ABC, Generic[T]):
    """
    API 응답 매퍼 기본 클래스
    
    모든 매퍼는 이 클래스를 상속받아 구현합니다.
    """
    
    def __init__(self, enable_cache: bool = True, cache_ttl_seconds: int = 60):
        """
        매퍼 초기화
        
        Args:
            enable_cache: 캐시 사용 여부
            cache_ttl_seconds: 캐시 유효 시간(초)
        """
        self.enable_cache = enable_cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = threading.RLock()  # 스레드 안전성을 위한 락
        
        logger.debug(f"{self.__class__.__name__} 초기화 완료")
    
    @abstractmethod
    def map_single(self, api_response: ApiResponse) -> T:
        """
        단일 API 응답을 도메인 엔터티로 변환
        
        Args:
            api_response: API 응답 데이터
            
        Returns:
            T: 변환된 도메인 엔터티
            
        Raises:
            MappingError: 매핑 실패 시
        """
        pass
    
    def map_list(self, api_response_list: List[ApiResponse]) -> List[T]:
        """
        API 응답 리스트를 도메인 엔터티 리스트로 변환
        
        Args:
            api_response_list: API 응답 데이터 리스트
            
        Returns:
            List[T]: 변환된 도메인 엔터티 리스트
        """
        try:
            entities = []
            for i, response in enumerate(api_response_list):
                try:
                    entity = self.map_single(response)
                    entities.append(entity)
                except MappingError as e:
                    logger.warning(f"리스트 항목 {i} 매핑 실패: {e}")
                    continue
            
            logger.debug(f"{len(entities)}/{len(api_response_list)} 항목 매핑 성공")
            return entities
            
        except Exception as e:
            logger.error(f"리스트 매핑 실패: {e}")
            raise MappingError(f"리스트 매핑 실패: {e}", api_response_list)
    
    def map_with_cache(self, api_response: ApiResponse, cache_key: Optional[str] = None) -> T:
        """
        캐시를 사용한 매핑
        
        Args:
            api_response: API 응답 데이터
            cache_key: 캐시 키 (없으면 자동 생성)
            
        Returns:
            T: 변환된 도메인 엔터티
        """
        if not self.enable_cache:
            return self.map_single(api_response)
        
        # 캐시 키 생성
        if cache_key is None:
            cache_key = self._generate_cache_key(api_response)
        
        # 캐시에서 조회
        with self._cache_lock:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                logger.debug(f"캐시에서 조회 성공: {cache_key}")
                return cached_result
            
            self._cache_misses += 1
        
        # 매핑 수행
        entity = self.map_single(api_response)
        
        # 캐시에 저장
        with self._cache_lock:
            self._save_to_cache(cache_key, entity)
        
        return entity
    
    def validate_api_response(self, api_response: ApiResponse, required_fields: List[str]) -> None:
        """
        API 응답 데이터 검증
        
        Args:
            api_response: API 응답 데이터
            required_fields: 필수 필드 목록
            
        Raises:
            MappingError: 검증 실패 시
        """
        if not isinstance(api_response, dict):
            raise MappingError(f"API 응답이 딕셔너리가 아닙니다: {type(api_response)}", api_response)
        
        missing_fields = []
        for field in required_fields:
            if field not in api_response:
                missing_fields.append(field)
        
        if missing_fields:
            raise MappingError(f"필수 필드 누락: {missing_fields}", api_response)
    
    def safe_get_value(self, api_response: ApiResponse, key: str, default: Any = None, 
                      value_type: type = str) -> Any:
        """
        안전한 값 추출
        
        Args:
            api_response: API 응답 데이터
            key: 추출할 키
            default: 기본값
            value_type: 예상 타입
            
        Returns:
            Any: 추출된 값
        """
        try:
            value = api_response.get(key, default)
            
            if value is None:
                return default
            
            # 문자열에서 쉼표 제거 (숫자 필드용)
            if value_type in [int, float] and isinstance(value, str):
                value = value.replace(',', '')
            
            # 타입 변환
            if value_type == int:
                return int(float(value)) if value else 0
            elif value_type == float:
                return float(value) if value else 0.0
            elif value_type == str:
                return str(value) if value else ""
            else:
                return value
                
        except (ValueError, TypeError) as e:
            logger.warning(f"값 변환 실패: {key}={value}, 타입={value_type}, 에러={e}")
            return default
    
    def _generate_cache_key(self, api_response: ApiResponse) -> str:
        """캐시 키 생성"""
        try:
            # API 응답을 JSON 문자열로 변환하여 해시 생성
            json_str = json.dumps(api_response, sort_keys=True, ensure_ascii=False)
            return hashlib.md5(json_str.encode('utf-8')).hexdigest()
        except Exception:
            # 해시 생성 실패 시 타임스탬프 기반 키 사용
            return f"fallback_{datetime.now().timestamp()}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[T]:
        """캐시에서 조회"""
        if cache_key not in self._cache:
            return None
        
        cache_entry = self._cache[cache_key]
        
        # TTL 확인
        if datetime.now() > cache_entry['expires_at']:
            del self._cache[cache_key]
            return None
        
        return cache_entry['entity']
    
    def _save_to_cache(self, cache_key: str, entity: T) -> None:
        """캐시에 저장"""
        expires_at = datetime.now() + timedelta(seconds=self.cache_ttl_seconds)
        
        self._cache[cache_key] = {
            'entity': entity,
            'expires_at': expires_at,
            'created_at': datetime.now()
        }
        
        # 캐시 정리 (100개 초과 시)
        if len(self._cache) > 100:
            self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """만료된 캐시 엔트리 정리"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        logger.debug(f"캐시 정리 완료: {len(expired_keys)}개 엔트리 제거")
    
    def clear_cache(self) -> None:
        """캐시 전체 삭제"""
        with self._cache_lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
        logger.debug("캐시 전체 삭제 완료")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        with self._cache_lock:
            now = datetime.now()
            active_entries = sum(1 for entry in self._cache.values() if now <= entry['expires_at'])
            
            total_requests = self._cache_hits + self._cache_misses
            hit_ratio = self._cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'total_entries': len(self._cache),
                'active_entries': active_entries,
                'expired_entries': len(self._cache) - active_entries,
                'cache_enabled': self.enable_cache,
                'ttl_seconds': self.cache_ttl_seconds,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'total_requests': total_requests,
                'hit_ratio': hit_ratio
            } 