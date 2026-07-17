---
name: programming-python
description: Python 3.8+ patterns — PEP 8, mandatory type hints, dependency-injectable testable design
when_to_use: Writing/modifying *.py files. User says 'implement in Python', 'add Python function/class'. SKIP for Python tests (testing-pytest), C++ (programming-cpp), CMake (programming-cmake).
paths: "**/*.py"
---

# Python Programming Skill

Use this skill when writing or modifying Python code.

<IMPORTANT>
Follow [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/) as the primary reference for style and conventions.

**Python 3.8+ Standard.** This project uses modern Python (3.8 or later). Use type hints, dataclasses, and modern features available in Python 3.8+.

**Type hints are REQUIRED.** All functions, methods, and class attributes must have type annotations. Use mypy for static type checking.

**Readability counts.** Python emphasizes clear, readable code. Follow the Zen of Python (PEP 20): "Explicit is better than implicit", "Simple is better than complex".

**All code MUST be unit testable.** Use dependency injection, avoid global state, and design for testability from the start.

**Guidelines override existing code style.** If you see existing code that violates the rules in this skill, apply these guidelines first and ignore the current project style. Do NOT propagate bad patterns just because they exist in the codebase.
</IMPORTANT>

## The Zen of Python (PEP 20)

Key principles to guide Python development:

- **Beautiful is better than ugly**
- **Explicit is better than implicit**
- **Simple is better than complex**
- **Complex is better than complicated**
- **Flat is better than nested**
- **Sparse is better than dense**
- **Readability counts**
- **Special cases aren't special enough to break the rules**
- **Errors should never pass silently**
- **In the face of ambiguity, refuse the temptation to guess**

## Code Style (PEP 8)

### Naming Conventions

```python
# Modules and packages: lowercase with underscores
import json_parser
from utils.data_processing import clean_data

# Classes: PascalCase
class UserAccount:
    pass

class HTTPConnectionPool:
    pass

# Functions and variables: snake_case
def calculate_total_price(items: list[Item]) -> float:
    total_amount = sum(item.price for item in items)
    return total_amount

# Constants: SCREAMING_SNAKE_CASE
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30

# Private: leading underscore
class Database:
    def _internal_method(self) -> None:
        pass

    def __very_private(self) -> None:  # Name mangling
        pass

# Protected (convention): single leading underscore
_module_level_private = "hidden"
```

### Indentation and Whitespace

```python
# Use 4 spaces per indentation level (NEVER tabs)
def long_function_name(
    var_one: str,
    var_two: int,
    var_three: dict[str, Any],
) -> bool:
    """Hanging indent for function arguments."""
    print(var_one)
    return True

# Line length: max 79 characters for code, 72 for docstrings/comments
# Break long lines at logical points
result = some_function_that_takes_arguments(
    argument1, argument2, argument3,
    argument4, argument5
)

# Two blank lines before top-level classes and functions
class MyClass:
    pass


def my_function() -> None:
    pass


# One blank line between methods
class Example:
    def method_one(self) -> None:
        pass

    def method_two(self) -> None:
        pass

# Whitespace in expressions
# GOOD
spam(ham[1], {eggs: 2})
if x == 4:
    print(x, y)
x, y = y, x

# BAD
spam( ham[ 1 ], { eggs: 2 } )
if x == 4 :
    print(x , y)
x , y = y , x
```

### Imports

```python
# Standard library imports first, then third-party, then local
# Each group separated by blank line, alphabetically sorted
import os
import sys
from typing import Any, Optional

import numpy as np
import requests

from myproject.utils import helper
from myproject.models import User

# Avoid wildcard imports
from module import *  # BAD

from module import specific_function  # GOOD

# One import per line for regular imports
import os
import sys

# Multiple items OK for 'from' imports
from typing import Any, Dict, List, Optional
```

## Type Hints (REQUIRED)

### Basic Type Hints

```python
from typing import Any, Optional, Union
from collections.abc import Sequence, Mapping, Callable

# Variables
name: str = "Alice"
age: int = 30
is_active: bool = True
scores: list[int] = [95, 87, 91]
user_data: dict[str, Any] = {"name": "Alice", "age": 30}

# Functions
def greet(name: str) -> str:
    return f"Hello, {name}!"

def process_data(
    items: list[str],
    batch_size: int = 10,
    validate: bool = True,
) -> dict[str, int]:
    """Process items and return statistics."""
    return {"processed": len(items), "batch_size": batch_size}

# Optional values (can be None)
def find_user(user_id: int) -> Optional[User]:
    """Returns User if found, None otherwise."""
    return users.get(user_id)

# Union types (Python 3.10+: use | instead)
def parse_input(value: Union[str, int]) -> int:
    """Accept string or int, return int."""
    return int(value)

# Python 3.10+ union syntax (preferred if available)
def parse_input(value: str | int) -> int:
    return int(value)
```

### Advanced Type Hints

```python
from typing import TypeVar, Generic, Protocol, Literal
from collections.abc import Callable, Iterator, Iterable

# Callable types
Callback = Callable[[str, int], bool]

def register_handler(callback: Callback) -> None:
    pass

# Generic types
T = TypeVar('T')

def first(items: list[T]) -> Optional[T]:
    return items[0] if items else None

# Generic classes
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

# Protocol (structural subtyping / duck typing)
class Drawable(Protocol):
    def draw(self) -> None:
        ...

def render(obj: Drawable) -> None:
    obj.draw()  # Any object with draw() method works

# Literal types
def set_mode(mode: Literal["read", "write", "append"]) -> None:
    pass

# Type aliases
UserId = int
UserMap = dict[UserId, User]

users: UserMap = {1: User("Alice"), 2: User("Bob")}
```

### Type Checking with mypy

```python
# Run mypy to check types
# $ mypy your_module.py

# Type ignore comments (use sparingly)
result = legacy_function()  # type: ignore[no-untyped-call]

# reveal_type for debugging (mypy only)
reveal_type(some_variable)  # mypy will show the inferred type
```

## Modern Python Features (3.8+)

### f-strings (Formatted String Literals)

```python
# GOOD: f-strings (readable, fast)
name = "Alice"
age = 30
message = f"Hello, {name}! You are {age} years old."
formatted = f"Result: {value:.2f}"  # With formatting
debug = f"{variable=}"  # Python 3.8+ debug syntax

# BAD: old-style formatting
message = "Hello, %s! You are %d years old." % (name, age)
message = "Hello, {}! You are {} years old.".format(name, age)
```

### Dataclasses

```python
from dataclasses import dataclass, field

# Simple dataclass
@dataclass
class Point:
    x: float
    y: float

    def distance(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5

p = Point(3.0, 4.0)  # Auto-generated __init__
print(p)  # Auto-generated __repr__

# With default values and field options
@dataclass
class User:
    username: str
    email: str
    active: bool = True
    roles: list[str] = field(default_factory=list)  # Mutable defaults
    _internal_id: int = field(default=0, repr=False, compare=False)

# Frozen (immutable) dataclass
@dataclass(frozen=True)
class Config:
    host: str
    port: int
    timeout: float = 30.0
```

### Walrus Operator (:=) - Python 3.8+

```python
# GOOD: Assign and use in one expression
if (match := pattern.search(text)) is not None:
    print(match.group(0))

# GOOD: In list comprehensions
filtered = [y for x in data if (y := transform(x)) is not None]

# GOOD: In while loops
while (line := file.readline()) != "":
    process(line)
```

### Pattern Matching (Python 3.10+)

```python
def process_command(command: dict[str, Any]) -> str:
    match command:
        case {"action": "create", "item": item}:
            return f"Creating {item}"
        case {"action": "delete", "id": user_id}:
            return f"Deleting user {user_id}"
        case {"action": "update", "id": user_id, "data": data}:
            return f"Updating {user_id} with {data}"
        case _:
            return "Unknown command"
```

### Context Managers

```python
# Built-in context managers
with open("file.txt", "r") as f:
    content = f.read()

# Custom context manager
from contextlib import contextmanager
from typing import Iterator

@contextmanager
def database_transaction(db: Database) -> Iterator[None]:
    """Context manager for database transactions."""
    db.begin()
    try:
        yield
        db.commit()
    except Exception:
        db.rollback()
        raise

# Usage
with database_transaction(db):
    db.execute("INSERT INTO users ...")
    db.execute("UPDATE accounts ...")
```

### Generators and Iterators

```python
from collections.abc import Iterator

# Generator function
def fibonacci(n: int) -> Iterator[int]:
    """Generate first n Fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Generator expression (memory efficient)
squares = (x**2 for x in range(1000000))  # Lazy evaluation
sum_of_squares = sum(x**2 for x in range(1000))

# Avoid creating unnecessary lists
# BAD
total = sum([x**2 for x in range(1000)])  # Creates intermediate list

# GOOD
total = sum(x**2 for x in range(1000))  # Generator expression
```

## Error Handling

### Exceptions

```python
# GOOD: Specific exceptions
try:
    result = risky_operation()
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid value: {e}")
    return default_value
finally:
    cleanup()

# BAD: Bare except
try:
    risky_operation()
except:  # Catches everything, including KeyboardInterrupt!
    pass

# GOOD: Specific exception catching
try:
    process()
except (ValueError, TypeError) as e:
    handle_error(e)

# Custom exceptions
class ValidationError(ValueError):
    """Raised when validation fails."""
    pass

class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code

# Raising exceptions
def validate_age(age: int) -> None:
    if age < 0:
        raise ValidationError(f"Age cannot be negative: {age}")
    if age > 150:
        raise ValidationError(f"Age is unrealistic: {age}")
```

### Exception Chaining

```python
# Chain exceptions to preserve context
try:
    result = parse_json(data)
except json.JSONDecodeError as e:
    raise ValidationError("Invalid JSON data") from e

# Suppress chaining if irrelevant
try:
    result = alternative_parser(data)
except Exception:
    raise ValidationError("Parsing failed") from None
```

### EAFP vs LBYL

```python
# EAFP: Easier to Ask Forgiveness than Permission (Pythonic)
try:
    value = dictionary[key]
except KeyError:
    value = default

# LBYL: Look Before You Leap (not Pythonic)
if key in dictionary:
    value = dictionary[key]
else:
    value = default

# EAFP is preferred in Python for:
# 1. Better performance (one lookup vs two)
# 2. Race condition safety
# 3. More readable for typical case
```

## Docstrings

### Google Style (Recommended)

```python
def calculate_distance(
    point1: tuple[float, float],
    point2: tuple[float, float],
    metric: str = "euclidean",
) -> float:
    """Calculate distance between two points.

    Computes the distance between two 2D points using the specified
    distance metric.

    Args:
        point1: First point as (x, y) coordinates.
        point2: Second point as (x, y) coordinates.
        metric: Distance metric to use. Options: "euclidean", "manhattan".
            Defaults to "euclidean".

    Returns:
        The calculated distance as a float.

    Raises:
        ValueError: If metric is not supported.

    Examples:
        >>> calculate_distance((0, 0), (3, 4))
        5.0
        >>> calculate_distance((0, 0), (3, 4), metric="manhattan")
        7.0
    """
    if metric == "euclidean":
        return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2) ** 0.5
    elif metric == "manhattan":
        return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])
    else:
        raise ValueError(f"Unsupported metric: {metric}")


class UserRepository:
    """Repository for managing user data.

    This class provides methods for creating, reading, updating, and
    deleting user records from the database.

    Attributes:
        db: Database connection instance.
        cache: Optional cache for frequently accessed users.
    """

    def __init__(self, db: Database, cache: Optional[Cache] = None) -> None:
        """Initialize the repository.

        Args:
            db: Database connection to use.
            cache: Optional cache instance for performance optimization.
        """
        self.db = db
        self.cache = cache
```

### One-Line Docstrings

```python
def get_username(user_id: int) -> str:
    """Return the username for the given user ID."""
    return users[user_id].username
```

### Inline Comments

**IMPORTANT: Avoid meaningless comments.** Only add inline comments when the code's intent is not self-evident. Python emphasizes readability - write self-documenting code with clear names and structure.

```python
# ❌ BAD - restates the obvious
count = 0  # Initialize count to zero
total = sum(numbers)  # Sum the numbers
result.append(item)  # Append item to result
user_list = []  # Empty list of users

# ❌ BAD - obvious operations
for user in users:  # Loop through users
    user.save()  # Save each user

# ❌ BAD - explaining standard Python features
data = [x for x in items if x > 0]  # List comprehension to filter positive values

# ✅ GOOD - explains why, not what
timeout *= 2  # Exponential backoff for retries
buffer_size = 8192  # Optimal size for network I/O on this system

# ✅ GOOD - explains non-obvious business logic
discount = 0.15 if is_vip else 0.05  # VIP customers get 15% discount per policy
offset = 3  # Skip magic bytes in file header

# ✅ GOOD - documents workarounds
time.sleep(0.1)  # Rate limit: API allows 10 requests/second
result = data.get("value", None)  # API sometimes omits this field

# ✅ GOOD - clarifies complex algorithms
# Binary search requires sorted input
idx = bisect.bisect_left(sorted_items, target)
```

**When to add comments:**

- Explaining **why** decisions were made (design rationale, business rules)
- Clarifying non-obvious algorithms or complex logic
- Documenting workarounds or known limitations
- Noting assumptions or preconditions
- Explaining performance optimizations
- TODOs or FIXMEs (sparingly, with ticket numbers)

**When NOT to add comments:**

- Describing what the code obviously does
- Repeating variable or function names
- Explaining basic Python syntax or standard library usage
- Commenting every line or obvious operations
- Adding docstring-style comments for internal/private functions (use docstrings instead)

## Performance Best Practices

### List Comprehensions and Generator Expressions

```python
# GOOD: List comprehension (readable, fast)
squares = [x**2 for x in range(10)]

# BAD: Loop with append (slower, less readable)
squares = []
for x in range(10):
    squares.append(x**2)

# GOOD: Generator for large datasets (memory efficient)
def process_large_file(filename: str) -> Iterator[str]:
    with open(filename) as f:
        for line in f:
            yield line.strip().upper()

# GOOD: Dict/set comprehensions
user_map = {user.id: user.name for user in users}
active_ids = {user.id for user in users if user.active}
```

### Use Built-in Functions

```python
# GOOD: Use built-ins (implemented in C, fast)
total = sum(numbers)
maximum = max(numbers)
minimum = min(numbers)
sorted_items = sorted(items, key=lambda x: x.priority)

# Use any() and all()
has_active = any(user.active for user in users)
all_valid = all(item.is_valid() for item in items)

# BAD: Manual loops for what built-ins can do
total = 0
for num in numbers:
    total += num
```

### String Concatenation

```python
# GOOD: Join for multiple strings
parts = ["Hello", "world", "from", "Python"]
message = " ".join(parts)

# GOOD: f-strings for formatting
result = f"User {name} has {count} items"

# BAD: Repeated concatenation in loop
message = ""
for part in parts:
    message += part + " "  # Creates new string each iteration

# GOOD: List + join for loops
parts_list = []
for item in items:
    parts_list.append(str(item))
message = ", ".join(parts_list)
```

### Avoid Repeated Lookups

```python
# BAD: Repeated method lookup
for item in items:
    results.append(item)  # Looks up 'append' every iteration

# GOOD: Cache the method
append = results.append
for item in items:
    append(item)

# BAD: Repeated attribute access
for i in range(len(container.items)):
    process(container.items[i])

# GOOD: Cache the attribute
items = container.items
for i in range(len(items)):
    process(items[i])

# BETTER: Direct iteration
for item in container.items:
    process(item)
```

### Use Local Variables

```python
# Local variables are faster than global lookups
import math

# BAD: Global lookup every iteration
def process_values(values: list[float]) -> list[float]:
    return [math.sqrt(x) for x in values]

# GOOD: Local variable
def process_values(values: list[float]) -> list[float]:
    sqrt = math.sqrt
    return [sqrt(x) for x in values]
```

## Testability (REQUIRED)

### Dependency Injection

```python
# BAD: Hard to test - creates its own dependencies
class OrderProcessor:
    def __init__(self) -> None:
        self.db = Database()  # Hard-coded dependency
        self.email = EmailService()

    def process(self, order: Order) -> None:
        self.db.save(order)
        self.email.send_confirmation(order)

# GOOD: Testable - dependencies injected
from typing import Protocol

class DatabaseProtocol(Protocol):
    def save(self, order: Order) -> None:
        ...

class EmailProtocol(Protocol):
    def send_confirmation(self, order: Order) -> None:
        ...

class OrderProcessor:
    def __init__(
        self,
        db: DatabaseProtocol,
        email: EmailProtocol,
    ) -> None:
        self.db = db
        self.email = email

    def process(self, order: Order) -> None:
        self.db.save(order)
        self.email.send_confirmation(order)

# Test with mocks
def test_order_processor():
    mock_db = Mock(spec=DatabaseProtocol)
    mock_email = Mock(spec=EmailProtocol)
    processor = OrderProcessor(mock_db, mock_email)

    order = Order(id=1, total=100)
    processor.process(order)

    mock_db.save.assert_called_once_with(order)
    mock_email.send_confirmation.assert_called_once_with(order)
```

### Pure Functions

```python
# GOOD: Pure function (testable, predictable)
def calculate_total(items: list[Item]) -> float:
    """Calculate total price of items."""
    return sum(item.price * item.quantity for item in items)

# BAD: Impure function (hard to test, unpredictable)
total_price = 0  # Global state

def add_to_total(item: Item) -> None:
    global total_price
    total_price += item.price * item.quantity
```

### Avoid Hidden Dependencies

```python
# BAD: Hidden dependencies (time, random, file system)
def create_user(name: str) -> User:
    return User(
        name=name,
        created_at=datetime.now(),  # Hidden dependency on system time
        id=random.randint(1, 1000000),  # Hidden dependency on random
    )

# GOOD: Explicit dependencies (injectable)
def create_user(
    name: str,
    created_at: datetime,
    id_generator: Callable[[], int],
) -> User:
    return User(
        name=name,
        created_at=created_at,
        id=id_generator(),
    )
```

## Code Organization

### Module Structure

```python
"""Module docstring explaining purpose.

This module provides functionality for processing user data
and generating reports.
"""

# Imports (grouped and sorted)
from __future__ import annotations  # For forward references

import os
import sys
from typing import Any, Optional

import requests
from sqlalchemy import create_engine

from myapp.models import User
from myapp.utils import logger

# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# Module-level variables (minimize these)
_cache: dict[str, Any] = {}

# Functions and classes
def process_user(user: User) -> dict[str, Any]:
    """Process user data."""
    pass


class UserProcessor:
    """Process and transform user data."""
    pass


# Main execution guard
if __name__ == "__main__":
    main()
```

### Package Structure

```text
myproject/
├── __init__.py
├── __main__.py          # Entry point: python -m myproject
├── config.py            # Configuration
├── models/
│   ├── __init__.py
│   ├── user.py
│   └── order.py
├── services/
│   ├── __init__.py
│   ├── database.py
│   └── email.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
└── tests/
    ├── __init__.py
    ├── test_models.py
    └── test_services.py
```

## Common Patterns

### Singleton Pattern (Use Sparingly)

```python
# GOOD: Module-level instance (simple singleton)
class DatabaseConnection:
    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string

    def connect(self) -> None:
        pass

# Create single instance at module level
db_connection = DatabaseConnection("postgresql://localhost/mydb")

# Better: Use dependency injection instead of singleton
```

### Factory Pattern

```python
from typing import Protocol

class Animal(Protocol):
    def speak(self) -> str:
        ...

class Dog:
    def speak(self) -> str:
        return "Woof!"

class Cat:
    def speak(self) -> str:
        return "Meow!"

def create_animal(animal_type: str) -> Animal:
    """Factory function for creating animals."""
    if animal_type == "dog":
        return Dog()
    elif animal_type == "cat":
        return Cat()
    else:
        raise ValueError(f"Unknown animal type: {animal_type}")
```

### Context Manager Pattern

```python
from typing import Any
from collections.abc import Iterator

class ResourceManager:
    """Manage a resource with automatic cleanup."""

    def __init__(self, resource_name: str) -> None:
        self.resource_name = resource_name
        self.resource: Any = None

    def __enter__(self) -> ResourceManager:
        """Acquire resource."""
        self.resource = acquire_resource(self.resource_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release resource."""
        if self.resource is not None:
            release_resource(self.resource)

# Usage
with ResourceManager("database") as rm:
    rm.resource.query("SELECT * FROM users")
```

## Anti-Patterns to Avoid

### Mutable Default Arguments

```python
# BAD: Mutable default argument
def add_item(item: str, items: list[str] = []) -> list[str]:
    items.append(item)
    return items

# GOOD: Use None and create new list
def add_item(item: str, items: Optional[list[str]] = None) -> list[str]:
    if items is None:
        items = []
    items.append(item)
    return items

# BETTER: For dataclasses, use field(default_factory=list)
from dataclasses import dataclass, field

@dataclass
class Container:
    items: list[str] = field(default_factory=list)
```

### Explicit Type Comparisons

```python
# BAD: Explicit type check
if type(value) == int:
    pass

# GOOD: isinstance (works with subclasses)
if isinstance(value, int):
    pass

# GOOD: Multiple types
if isinstance(value, (int, float)):
    pass
```

### Using `is` for Value Comparison

```python
# BAD: Using 'is' for value comparison
if name is "Alice":  # Compares identity, not value!
    pass

# GOOD: Using == for value comparison
if name == "Alice":
    pass

# OK: Using 'is' for None, True, False
if value is None:
    pass
```

## Code Style Checklist

Before submitting Python code:

### Type Hints & Documentation

- [ ] All functions have type hints for parameters and return values
- [ ] All public APIs have docstrings (Google style)
- [ ] Inline comments explain **why**, not what (avoid obvious/meaningless comments)
- [ ] Complex logic has explanatory comments only when non-obvious

### Style & Conventions

- [ ] Follows PEP 8 naming conventions (snake_case, PascalCase, etc.)
- [ ] Line length ≤ 79 characters
- [ ] Proper import organization (stdlib, third-party, local)
- [ ] No mutable default arguments
- [ ] Uses f-strings for string formatting

### Modern Python

- [ ] Uses dataclasses where appropriate
- [ ] Uses type hints (Python 3.8+ syntax)
- [ ] Uses context managers for resource management
- [ ] Avoids deprecated features (%, .format(), etc.)

### Error Handling

- [ ] Specific exception types (no bare `except:`)
- [ ] Custom exceptions inherit from appropriate base
- [ ] EAFP (try/except) over LBYL (if checks) where appropriate

### Performance

- [ ] List/dict/set comprehensions instead of loops where readable
- [ ] Generator expressions for large datasets
- [ ] Built-in functions used (sum, max, min, sorted, any, all)
- [ ] String joining with `str.join()`, not concatenation in loops

### Testability

- [ ] **Unit testable** - dependencies injected, not created internally
- [ ] No hidden dependencies (time, random, file system)
- [ ] Pure functions where possible (same input → same output)
- [ ] Protocols/interfaces for dependency injection

### Code Organization

- [ ] One class per file (generally, unless tightly coupled)
- [ ] Functions and classes ordered logically
- [ ] No unused imports or variables
- [ ] Clear module structure

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Python Type Checking (Guide)](https://realpython.com/python-type-checking/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [The Hitchhiker's Guide to Python](https://docs.python-guide.org/)
