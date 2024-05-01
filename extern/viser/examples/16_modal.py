"""Modal basics

Examples of using modals in Viser."""

import time

import viser


def main():
    server = viser.ViserServer()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        with client.add_gui_modal("Modal example"):
            client.add_gui_markdown(
                "**The input below determines the title of the modal...**"
            )

            gui_title = client.add_gui_text(
                "Title",
                initial_value="My Modal",
            )

            modal_button = client.add_gui_button("Show more modals")

            @modal_button.on_click
            def _(_) -> None:
                with client.add_gui_modal(gui_title.value) as modal:
                    client.add_gui_markdown("This is content inside the modal!")
                    client.add_gui_button("Close").on_click(lambda _: modal.close())

    while True:
        time.sleep(0.15)


if __name__ == "__main__":
    main()