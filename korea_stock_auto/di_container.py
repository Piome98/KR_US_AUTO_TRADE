"""
한국 주식 자동매매 - 의존성 주입 컨테이너

타입 안전한 의존성 주입 컨테이너로 모든 서비스 객체를 중앙 관리합니다.
테스트 시 Mock 객체 주입이 용이하며, 설정 변경 시 영향도를 최소화합니다.
"""

from typing import TypeVar, Type, Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DIContainerError(Exception):
    """의존성 주입 컨테이너 오류"""
    pass


@dataclass
class ServiceDefinition:
    """서비스 정의 클래스"""
    service_type: Type
    factory: Callable[[], Any]
    singleton: bool = True
    instance: Optional[Any] = None


class DIContainer:
    """의존성 주입 컨테이너"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDefinition] = {}
        self._instances: Dict[Type, Any] = {}
        
    def register_singleton(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """싱글톤 서비스 등록"""
        self._services[service_type] = ServiceDefinition(
            service_type=service_type,
            factory=factory,
            singleton=True
        )
        logger.debug(f"싱글톤 서비스 등록: {service_type.__name__}")
    
    def register_transient(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """트랜지언트 서비스 등록 (매번 새 인스턴스)"""
        self._services[service_type] = ServiceDefinition(
            service_type=service_type,
            factory=factory,
            singleton=False
        )
        logger.debug(f"트랜지언트 서비스 등록: {service_type.__name__}")
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """기존 인스턴스 등록"""
        self._instances[service_type] = instance
        logger.debug(f"인스턴스 등록: {service_type.__name__}")
    
    def get(self, service_type: Type[T]) -> T:
        """서비스 인스턴스 가져오기"""
        # 등록된 인스턴스가 있으면 우선 반환
        if service_type in self._instances:
            return self._instances[service_type]
        
        # 서비스 정의가 없으면 오류
        if service_type not in self._services:
            raise DIContainerError(f"서비스가 등록되지 않음: {service_type.__name__}")
        
        service_def = self._services[service_type]
        
        # 싱글톤이고 이미 생성된 인스턴스가 있으면 반환
        if service_def.singleton and service_def.instance is not None:
            return service_def.instance
        
        # 새 인스턴스 생성
        try:
            instance = service_def.factory()
            
            # 싱글톤이면 저장
            if service_def.singleton:
                service_def.instance = instance
            
            logger.debug(f"서비스 인스턴스 생성: {service_type.__name__}")
            return instance
            
        except Exception as e:
            raise DIContainerError(f"서비스 생성 실패 {service_type.__name__}: {str(e)}")
    
    def clear(self) -> None:
        """모든 서비스와 인스턴스 초기화"""
        self._services.clear()
        self._instances.clear()
        logger.debug("DI 컨테이너 초기화 완료")
    
    def is_registered(self, service_type: Type) -> bool:
        """서비스 등록 여부 확인"""
        return service_type in self._services or service_type in self._instances


# 전역 DI 컨테이너 인스턴스
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """전역 DI 컨테이너 가져오기"""
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


def reset_container() -> None:
    """전역 DI 컨테이너 초기화 (주로 테스트용)"""
    global _container
    _container = None 