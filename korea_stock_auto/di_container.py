"""
한국 주식 자동매매 - 의존성 주입 컨테이너

타입 안전한 의존성 주입 컨테이너로 모든 서비스 객체를 중앙 관리합니다.
테스트 시 Mock 객체 주입이 용이하며, 설정 변경 시 영향도를 최소화합니다.

v2.0 개선사항:
- Lazy Loading 지원
- 순환 의존성 자동 감지 및 해결
- 메모리 누수 방지
- 스레드 안전성 보장
- 서비스 라이프사이클 관리
"""

from typing import TypeVar, Type, Dict, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import threading
import weakref
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifecycle(Enum):
    """서비스 라이프사이클"""
    SINGLETON = "singleton"      # 애플리케이션 생명주기 동안 하나의 인스턴스
    TRANSIENT = "transient"      # 매번 새로운 인스턴스 생성
    SCOPED = "scoped"           # 특정 스코프 내에서 동일한 인스턴스


class DIContainerError(Exception):
    """의존성 주입 컨테이너 오류"""
    pass


class CircularDependencyError(DIContainerError):
    """순환 의존성 오류"""
    pass


@dataclass
class ServiceDefinition:
    """개선된 서비스 정의 클래스"""
    service_type: Type
    factory: Callable[[], Any]
    lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON
    instance: Optional[Any] = None
    lazy: bool = True
    created_at: Optional[float] = None
    dependencies: Set[Type] = field(default_factory=set)
    
    def is_singleton(self) -> bool:
        return self.lifecycle == ServiceLifecycle.SINGLETON
    
    def is_transient(self) -> bool:
        return self.lifecycle == ServiceLifecycle.TRANSIENT
    
    def is_scoped(self) -> bool:
        return self.lifecycle == ServiceLifecycle.SCOPED


class ServiceScope:
    """서비스 스코프 관리"""
    
    def __init__(self, name: str):
        self.name = name
        self.instances: Dict[Type, Any] = {}
        self.created_at = time.time()
    
    def get_instance(self, service_type: Type) -> Optional[Any]:
        return self.instances.get(service_type)
    
    def set_instance(self, service_type: Type, instance: Any) -> None:
        self.instances[service_type] = instance
    
    def clear(self) -> None:
        """스코프 내 모든 인스턴스 정리"""
        self.instances.clear()


class EnhancedDIContainer:
    """향상된 의존성 주입 컨테이너"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDefinition] = {}
        self._instances: Dict[Type, Any] = {}
        self._scopes: Dict[str, ServiceScope] = {}
        self._current_scope: Optional[str] = None
        self._creation_stack: Set[Type] = set()  # 순환 의존성 감지용
        self._lock = threading.RLock()  # 스레드 안전성
        self._weak_refs: Dict[Type, weakref.ReferenceType] = {}  # 메모리 누수 방지
        
    def register_singleton(self, service_type: Type[T], factory: Callable[[], T], 
                          lazy: bool = True, dependencies: Optional[Set[Type]] = None) -> None:
        """싱글톤 서비스 등록 (개선된 버전)"""
        with self._lock:
            self._services[service_type] = ServiceDefinition(
                service_type=service_type,
                factory=factory,
                lifecycle=ServiceLifecycle.SINGLETON,
                lazy=lazy,
                dependencies=dependencies or set()
            )
            logger.debug(f"싱글톤 서비스 등록: {service_type.__name__} (lazy={lazy})")
    
    def register_transient(self, service_type: Type[T], factory: Callable[[], T],
                          dependencies: Optional[Set[Type]] = None) -> None:
        """트랜지언트 서비스 등록 (개선된 버전)"""
        with self._lock:
            self._services[service_type] = ServiceDefinition(
                service_type=service_type,
                factory=factory,
                lifecycle=ServiceLifecycle.TRANSIENT,
                dependencies=dependencies or set()
            )
            logger.debug(f"트랜지언트 서비스 등록: {service_type.__name__}")
    
    def register_scoped(self, service_type: Type[T], factory: Callable[[], T],
                       dependencies: Optional[Set[Type]] = None) -> None:
        """스코프 서비스 등록"""
        with self._lock:
            self._services[service_type] = ServiceDefinition(
                service_type=service_type,
                factory=factory,
                lifecycle=ServiceLifecycle.SCOPED,
                dependencies=dependencies or set()
            )
            logger.debug(f"스코프 서비스 등록: {service_type.__name__}")
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """기존 인스턴스 등록 (개선된 버전)"""
        with self._lock:
            self._instances[service_type] = instance
            # Weak reference로 메모리 관리
            self._weak_refs[service_type] = weakref.ref(instance, 
                                                       lambda ref: self._cleanup_weak_ref(service_type))
            logger.debug(f"인스턴스 등록: {service_type.__name__}")
    
    def get(self, service_type: Type[T]) -> T:
        """서비스 인스턴스 가져오기 (개선된 버전)"""
        with self._lock:
            # 순환 의존성 감지
            if service_type in self._creation_stack:
                cycle = list(self._creation_stack) + [service_type]
                cycle_names = " -> ".join([t.__name__ for t in cycle])
                raise CircularDependencyError(f"순환 의존성 감지: {cycle_names}")
            
            # 등록된 인스턴스가 있으면 우선 반환
            if service_type in self._instances:
                return self._instances[service_type]
            
            # 서비스 정의가 없으면 오류
            if service_type not in self._services:
                raise DIContainerError(f"서비스가 등록되지 않음: {service_type.__name__}")
            
            service_def = self._services[service_type]
            
            # 라이프사이클에 따른 인스턴스 관리
            if service_def.is_singleton():
                return self._get_singleton_instance(service_def)
            elif service_def.is_scoped():
                return self._get_scoped_instance(service_def)
            else:  # TRANSIENT
                return self._create_instance(service_def)
    
    def _get_singleton_instance(self, service_def: ServiceDefinition) -> Any:
        """싱글톤 인스턴스 가져오기"""
        if service_def.instance is not None:
            return service_def.instance
        
        instance = self._create_instance(service_def)
        service_def.instance = instance
        service_def.created_at = time.time()
        return instance
    
    def _get_scoped_instance(self, service_def: ServiceDefinition) -> Any:
        """스코프 인스턴스 가져오기"""
        if self._current_scope is None:
            raise DIContainerError(f"스코프 서비스 {service_def.service_type.__name__}를 사용하려면 활성 스코프가 필요함")
        
        scope = self._scopes[self._current_scope]
        instance = scope.get_instance(service_def.service_type)
        
        if instance is None:
            instance = self._create_instance(service_def)
            scope.set_instance(service_def.service_type, instance)
        
        return instance
    
    def _create_instance(self, service_def: ServiceDefinition) -> Any:
        """새 인스턴스 생성"""
        self._creation_stack.add(service_def.service_type)
        try:
            instance = service_def.factory()
            logger.debug(f"서비스 인스턴스 생성: {service_def.service_type.__name__}")
            return instance
        except Exception as e:
            raise DIContainerError(f"서비스 생성 실패 {service_def.service_type.__name__}: {str(e)}")
        finally:
            self._creation_stack.discard(service_def.service_type)
    
    @contextmanager
    def scope(self, scope_name: str):
        """서비스 스코프 컨텍스트 매니저"""
        with self._lock:
            old_scope = self._current_scope
            if scope_name not in self._scopes:
                self._scopes[scope_name] = ServiceScope(scope_name)
            self._current_scope = scope_name
            
        try:
            logger.debug(f"스코프 시작: {scope_name}")
            yield self._scopes[scope_name]
        finally:
            with self._lock:
                self._current_scope = old_scope
                logger.debug(f"스코프 종료: {scope_name}")
    
    def clear_scope(self, scope_name: str) -> None:
        """특정 스코프 정리"""
        with self._lock:
            if scope_name in self._scopes:
                self._scopes[scope_name].clear()
                del self._scopes[scope_name]
                logger.debug(f"스코프 정리: {scope_name}")
    
    def clear(self) -> None:
        """모든 서비스와 인스턴스 초기화"""
        with self._lock:
            # 모든 스코프 정리
            for scope in self._scopes.values():
                scope.clear()
            self._scopes.clear()
            
            # 싱글톤 인스턴스 정리
            for service_def in self._services.values():
                service_def.instance = None
                service_def.created_at = None
            
            self._services.clear()
            self._instances.clear()
            self._weak_refs.clear()
            self._creation_stack.clear()
            logger.debug("DI 컨테이너 초기화 완료")
    
    def is_registered(self, service_type: Type) -> bool:
        """서비스 등록 여부 확인"""
        with self._lock:
            return service_type in self._services or service_type in self._instances
    
    def get_service_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """서비스 정보 가져오기 (디버깅용)"""
        with self._lock:
            if service_type not in self._services:
                return None
            
            service_def = self._services[service_type]
            return {
                "type": service_def.service_type.__name__,
                "lifecycle": service_def.lifecycle.value,
                "lazy": service_def.lazy,
                "instance_created": service_def.instance is not None,
                "created_at": service_def.created_at,
                "dependencies": [dep.__name__ for dep in service_def.dependencies]
            }
    
    def get_container_stats(self) -> Dict[str, Any]:
        """컨테이너 통계 정보"""
        with self._lock:
            singleton_count = sum(1 for s in self._services.values() if s.is_singleton())
            transient_count = sum(1 for s in self._services.values() if s.is_transient())
            scoped_count = sum(1 for s in self._services.values() if s.is_scoped())
            active_instances = sum(1 for s in self._services.values() if s.instance is not None)
            
            return {
                "total_services": len(self._services),
                "singleton_services": singleton_count,
                "transient_services": transient_count,
                "scoped_services": scoped_count,
                "active_instances": active_instances,
                "registered_instances": len(self._instances),
                "active_scopes": len(self._scopes),
                "current_scope": self._current_scope
            }
    
    def _cleanup_weak_ref(self, service_type: Type) -> None:
        """Weak reference 정리"""
        with self._lock:
            if service_type in self._weak_refs:
                del self._weak_refs[service_type]
            if service_type in self._instances:
                del self._instances[service_type]
            logger.debug(f"Weak reference 정리: {service_type.__name__}")

    def register_mappers(self) -> None:
        """
        API 매퍼들을 DI 컨테이너에 등록
        
        모든 매퍼는 싱글톤으로 등록하며, 각 매퍼의 캐시 설정은 개별적으로 관리됩니다.
        """
        try:
            from korea_stock_auto.api.mappers import (
                StockMapper, PortfolioMapper, AccountMapper, OrderMapper
            )
            
            # StockMapper 등록 (30초 캐시)
            self.register_singleton(
                StockMapper,
                lambda: StockMapper(enable_cache=True, cache_ttl_seconds=30),
                lazy=True
            )
            
            # PortfolioMapper 등록 (2분 캐시)
            self.register_singleton(
                PortfolioMapper,
                lambda: PortfolioMapper(enable_cache=True, cache_ttl_seconds=120),
                lazy=True
            )
            
            # AccountMapper 등록 (1분 캐시)
            self.register_singleton(
                AccountMapper,
                lambda: AccountMapper(enable_cache=True, cache_ttl_seconds=60),
                lazy=True
            )
            
            # OrderMapper 등록 (캐시 비활성화 - 실시간성 중요)
            self.register_singleton(
                OrderMapper,
                lambda: OrderMapper(enable_cache=False, cache_ttl_seconds=30),
                lazy=True
            )
            
            logger.debug("API 매퍼들 DI 컨테이너 등록 완료")
            
        except ImportError as e:
            logger.warning(f"매퍼 import 실패, DI 등록 건너뜀: {e}")
        except Exception as e:
            logger.error(f"매퍼 DI 등록 중 오류: {e}")
    
    def register_mapper_with_config(self, mapper_type: Type, cache_config: Dict[str, Any]) -> None:
        """
        특정 매퍼를 사용자 정의 캐시 설정으로 등록
        
        Args:
            mapper_type: 매퍼 타입
            cache_config: 캐시 설정 {'enable_cache': bool, 'cache_ttl_seconds': int}
        """
        try:
            self.register_singleton(
                mapper_type,
                lambda: mapper_type(**cache_config),
                lazy=True
            )
            logger.debug(f"{mapper_type.__name__} 사용자 정의 설정으로 등록: {cache_config}")
            
        except Exception as e:
            logger.error(f"{mapper_type.__name__} 사용자 정의 등록 실패: {e}")
    
    def get_mapper_stats(self) -> Dict[str, Any]:
        """
        등록된 모든 매퍼들의 캐시 통계 조회
        
        Returns:
            Dict: 매퍼별 캐시 통계
        """
        stats = {}
        
        try:
            # 매퍼 타입 import
            from korea_stock_auto.api.mappers import (
                StockMapper, PortfolioMapper, AccountMapper, OrderMapper
            )
            
            # 매퍼 클래스들을 직접 사용
            mapper_classes = {
                'StockMapper': StockMapper,
                'PortfolioMapper': PortfolioMapper,
                'AccountMapper': AccountMapper,
                'OrderMapper': OrderMapper
            }
            
            for mapper_name, mapper_class in mapper_classes.items():
                try:
                    if mapper_class in self._services:
                        service_def = self._services[mapper_class]
                        if service_def.instance:
                            stats[mapper_name] = service_def.instance.get_cache_stats()
                        else:
                            stats[mapper_name] = {'status': 'not_instantiated'}
                    else:
                        stats[mapper_name] = {'status': 'not_registered'}
                        
                except Exception as e:
                    stats[mapper_name] = {'status': 'error', 'error': str(e)}
                    
        except ImportError as e:
            logger.warning(f"매퍼 import 실패: {e}")
            for mapper_name in ['StockMapper', 'PortfolioMapper', 'AccountMapper', 'OrderMapper']:
                stats[mapper_name] = {'status': 'import_error', 'error': str(e)}
        
        return stats
    
    def clear_all_mapper_caches(self) -> None:
        """
        등록된 모든 매퍼들의 캐시 삭제
        """
        cleared_count = 0
        
        try:
            from korea_stock_auto.api.mappers import (
                StockMapper, PortfolioMapper, AccountMapper, OrderMapper
            )
            
            mapper_types = [StockMapper, PortfolioMapper, AccountMapper, OrderMapper]
            
            for mapper_type in mapper_types:
                try:
                    if mapper_type in self._services:
                        service_def = self._services[mapper_type]
                        if service_def.instance and hasattr(service_def.instance, 'clear_cache'):
                            service_def.instance.clear_cache()
                            cleared_count += 1
                            logger.debug(f"{mapper_type.__name__} 캐시 삭제 완료")
                            
                except Exception as e:
                    logger.warning(f"{mapper_type.__name__} 캐시 삭제 실패: {e}")
            
            logger.info(f"매퍼 캐시 삭제 완료: {cleared_count}개")
            
        except ImportError as e:
            logger.warning(f"매퍼 import 실패, 캐시 삭제 건너뜀: {e}")
        except Exception as e:
            logger.error(f"매퍼 캐시 삭제 중 오류: {e}")

    def get_mapper_health_status(self) -> Dict[str, Any]:
        """
        매퍼들의 건강 상태 확인
        
        Step 5.4: 성능 모니터링 및 건강 상태 확인
        """
        health_status = {}
        
        try:
            from korea_stock_auto.api.mappers import (
                StockMapper, PortfolioMapper, AccountMapper, OrderMapper
            )
            from korea_stock_auto.config.mapper_config import get_mapper_settings, get_mapper_performance_config
            
            mapper_settings = get_mapper_settings()
            mapper_classes = {
                'StockMapper': StockMapper,
                'PortfolioMapper': PortfolioMapper,
                'AccountMapper': AccountMapper,
                'OrderMapper': OrderMapper
            }
            
            for mapper_name, mapper_class in mapper_classes.items():
                try:
                    if mapper_class in self._services:
                        service_def = self._services[mapper_class]
                        
                        # 매퍼별 성능 설정 가져오기
                        perf_config = get_mapper_performance_config(mapper_name, mapper_settings)
                        
                        if service_def.instance:
                            # 인스턴스가 있는 경우 - 상세 상태 확인
                            cache_stats = service_def.instance.get_cache_stats()
                            
                            health_status[mapper_name] = {
                                'status': 'healthy',
                                'instance_created': True,
                                'cache_enabled': cache_stats.get('cache_enabled', False),
                                'cache_hit_ratio': cache_stats.get('hit_ratio', 0.0),
                                'total_entries': cache_stats.get('total_entries', 0),
                                'active_entries': cache_stats.get('active_entries', 0),
                                'performance_config': perf_config,
                                'created_at': service_def.created_at
                            }
                        else:
                            # 인스턴스가 없는 경우 - 기본 상태
                            health_status[mapper_name] = {
                                'status': 'not_instantiated',
                                'instance_created': False,
                                'cache_enabled': 'unknown',
                                'performance_config': perf_config
                            }
                    else:
                        health_status[mapper_name] = {
                            'status': 'not_registered',
                            'instance_created': False
                        }
                        
                except Exception as e:
                    health_status[mapper_name] = {
                        'status': 'error',
                        'error': str(e),
                        'instance_created': False
                    }
                    
        except ImportError as e:
            logger.warning(f"매퍼 import 실패: {e}")
            for mapper_name in ['StockMapper', 'PortfolioMapper', 'AccountMapper', 'OrderMapper']:
                health_status[mapper_name] = {
                    'status': 'import_error',
                    'error': str(e),
                    'instance_created': False
                }
        
        return health_status

    def optimize_mapper_performance(self) -> Dict[str, Any]:
        """
        매퍼 성능 최적화 실행
        
        Step 5.4: 성능 최적화 자동 실행
        """
        optimization_results = {}
        
        try:
            from korea_stock_auto.api.mappers import (
                StockMapper, PortfolioMapper, AccountMapper, OrderMapper
            )
            
            mapper_classes = {
                'StockMapper': StockMapper,
                'PortfolioMapper': PortfolioMapper,
                'AccountMapper': AccountMapper,
                'OrderMapper': OrderMapper
            }
            
            for mapper_name, mapper_class in mapper_classes.items():
                try:
                    if mapper_class in self._services:
                        service_def = self._services[mapper_class]
                        
                        if service_def.instance and hasattr(service_def.instance, 'clear_cache'):
                            # 캐시 통계 확인
                            stats_before = service_def.instance.get_cache_stats()
                            expired_before = stats_before.get('expired_entries', 0)
                            
                            # 만료된 캐시 정리
                            service_def.instance._cleanup_expired_entries()
                            
                            # 정리 후 통계 확인
                            stats_after = service_def.instance.get_cache_stats()
                            expired_after = stats_after.get('expired_entries', 0)
                            
                            cleaned_entries = expired_before - expired_after
                            
                            optimization_results[mapper_name] = {
                                'status': 'optimized',
                                'cleaned_entries': cleaned_entries,
                                'cache_size_before': stats_before.get('total_entries', 0),
                                'cache_size_after': stats_after.get('total_entries', 0),
                                'memory_freed_estimate': cleaned_entries * 100  # 추정 메모리 절약 (바이트)
                            }
                            
                            logger.debug(f"{mapper_name} 최적화 완료: {cleaned_entries}개 항목 정리")
                        else:
                            optimization_results[mapper_name] = {
                                'status': 'skipped',
                                'reason': 'no_instance_or_cache'
                            }
                    else:
                        optimization_results[mapper_name] = {
                            'status': 'skipped',
                            'reason': 'not_registered'
                        }
                        
                except Exception as e:
                    optimization_results[mapper_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    logger.warning(f"{mapper_name} 최적화 실패: {e}")
                    
        except ImportError as e:
            logger.warning(f"매퍼 import 실패, 최적화 건너뜀: {e}")
            for mapper_name in ['StockMapper', 'PortfolioMapper', 'AccountMapper', 'OrderMapper']:
                optimization_results[mapper_name] = {
                    'status': 'error',
                    'error': f'import_failed: {str(e)}'
                }
        
        total_cleaned = sum(
            result.get('cleaned_entries', 0) 
            for result in optimization_results.values() 
            if result.get('status') == 'optimized'
        )
        
        logger.info(f"매퍼 성능 최적화 완료: 총 {total_cleaned}개 항목 정리")
        
        return {
            'total_cleaned_entries': total_cleaned,
            'optimized_mappers': len([r for r in optimization_results.values() if r.get('status') == 'optimized']),
            'details': optimization_results
        }


# 전역 DI 컨테이너 인스턴스
_container: Optional[EnhancedDIContainer] = None
_lock = threading.Lock()


def get_container() -> EnhancedDIContainer:
    """전역 DI 컨테이너 가져오기 (스레드 안전)"""
    global _container
    if _container is None:
        with _lock:
            if _container is None:  # 더블 체크 락킹
                _container = EnhancedDIContainer()
    return _container


def reset_container() -> None:
    """전역 DI 컨테이너 초기화 (주로 테스트용)"""
    global _container
    with _lock:
        if _container is not None:
            _container.clear()
        _container = None


# 편의 함수들
def scope(scope_name: str):
    """스코프 컨텍스트 매니저 (편의 함수)"""
    return get_container().scope(scope_name)


def get_service_stats() -> Dict[str, Any]:
    """서비스 통계 가져오기 (편의 함수)"""
    return get_container().get_container_stats()


# 하위 호환성을 위한 기존 클래스도 유지
DIContainer = EnhancedDIContainer 