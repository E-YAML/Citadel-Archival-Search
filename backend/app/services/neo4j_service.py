from typing import Optional
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import DriverError, Neo4jError
from loguru import logger

from app.core.config import settings


class Neo4jConnectionError(Exception):
    """Raised when connection verification or operations with Neo4j fail."""
    pass


class Neo4jService:
    """
    Service layer managing the lifecycle and health of the Neo4j Graph Database driver.
    """
    def __init__(self) -> None:
        self._driver: Optional[AsyncDriver] = None

    def get_driver(self) -> AsyncDriver:
        """
        Retrieves the singleton instance of the Neo4j AsyncDriver.
        Initializes it if not already instantiated.
        """
        if self._driver is None:
            try:
                # Instantiate driver asynchronously using Settings
                self._driver = AsyncGraphDatabase.driver(
                    uri=settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
                )
                logger.info("Neo4j AsyncDriver successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j AsyncDriver: {str(e)}")
                raise Neo4jConnectionError(f"Neo4j driver setup failed: {str(e)}") from e
        return self._driver

    async def verify_connectivity(self) -> bool:
        """
        Verifies connectivity to the Neo4j instance by calling the native driver's
        connectivity validation and executing a simple verification query ('RETURN 1').
        """
        driver = self.get_driver()
        try:
            logger.info("Verifying connectivity to Neo4j instance...")
            # Native verify_connectivity check
            await driver.verify_connectivity()
            
            # Execute verification dummy query inside a session context block
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS val")
                record = await result.single()
                if record and record["val"] == 1:
                    logger.info("Neo4j database connection and execution test succeeded.")
                    return True
            logger.error("Neo4j dummy query execution returned unexpected results.")
            return False
        except (DriverError, Neo4jError) as e:
            logger.error(f"Neo4j connectivity check encountered driver or query failure: {str(e)}")
            raise Neo4jConnectionError(f"Database error during connectivity test: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Neo4j connectivity verification: {str(e)}")
            raise Neo4jConnectionError(f"Unexpected connectivity error: {str(e)}") from e

    async def close(self) -> None:
        """
        Closes the active AsyncDriver session and connection pool safely.
        """
        if self._driver is not None:
            try:
                await self._driver.close()
                logger.info("Neo4j AsyncDriver closed.")
            except Exception as e:
                logger.error(f"Error while closing Neo4j AsyncDriver: {str(e)}")
            finally:
                self._driver = None
