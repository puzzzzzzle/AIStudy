import logging

import typer
from rich.console import Console
from rich.traceback import install
from rich import print as rich_print
from rich.logging import RichHandler

app = typer.Typer()


def do_something(info: str) -> None:
    raise NotImplementedError(f"{info}")


@app.command()
def hello(name: str):
    print(f"Hello {name}")


@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")


@app.command()
def test_traceback(info: str):
    install(show_locals=True)
    console = Console()
    try:
        do_something(info)
    except Exception:
        console.print_exception(show_locals=True)
@app.command()
def test_rich_print(info: str):
    rich_print(f"Hello, [bold magenta]World[/bold magenta]!", ":vampire:", locals())
    rich_print(f"info is {info}")


test_data = [
    {"jsonrpc": "2.0", "method": "sum", "params": [None, 1, 2, 4, False, True], "id": "1",},
    {"jsonrpc": "2.0", "method": "notify_hello", "params": [7]},
    {"jsonrpc": "2.0", "method": "subtract", "params": [42, 23], "id": "2"},
]
@app.command()
def test_log():
    console = Console()
    enabled = False
    context = {
        "foo": "bar",
    }

    movies = ["Deadpool", "Rise of the Skywalker"]
    console.log("Hello from", console, "!")
    console.log(test_data, log_locals=True)

@app.command()
def test_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True,
                              tracebacks_show_locals=True)]
    )
    logger = logging.getLogger("test_logging")
    logger.info(f"logger from rich")
    local_vars = {"foo": "bar", "num": 42}
    logger.info("带 locals 的日志", extra={"locals": local_vars})
    try:
        do_something("test logging")
    except Exception as e:
        logger.info(f"get error")
        logger.exception(e)
        # logger.exception(e)

if __name__ == "__main__":
    app()
