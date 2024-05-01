import { ColorTranslator } from "colortranslator";
import {
  GuiAddFolderMessage,
  GuiAddTabGroupMessage,
} from "../WebsocketMessages";
import { ViewerContext, ViewerContextContents } from "../App";
import { makeThrottledMessageSender } from "../WebsocketFunctions";
import { GuiConfig } from "./GuiState";
import {
  Collapse,
  Image,
  Paper,
  Tabs,
  TabsValue,
  useMantineTheme,
} from "@mantine/core";

import {
  Box,
  Button,
  Checkbox,
  ColorInput,
  Flex,
  NumberInput,
  Select,
  Slider,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import React from "react";
import Markdown from "../Markdown";
import { ErrorBoundary } from "react-error-boundary";
import { useDisclosure } from "@mantine/hooks";
import { IconChevronDown, IconChevronUp } from "@tabler/icons-react";

/** Root of generated inputs. */
export default function GeneratedGuiContainer({
  // We need to take viewer as input in drei's <Html /> elements, where contexts break.
  containerId,
  viewer,
}: {
  containerId: string;
  viewer?: ViewerContextContents;
}) {
  if (viewer === undefined) viewer = React.useContext(ViewerContext)!;

  const guiIdSet = viewer.useGui(
    (state) => state.guiIdSetFromContainerId[containerId],
  );
  const guiConfigFromId = viewer.useGui((state) => state.guiConfigFromId);

  // Render each GUI element in this container.
  const out =
    guiIdSet === undefined ? null : (
      <Box pt="xs">
        {[...guiIdSet]
          .map((id) => guiConfigFromId[id])
          .sort((a, b) => a.order - b.order)
          .map((conf, index) => {
            return (
              <>
                {conf.type === "GuiAddFolderMessage" && index > 0 ? (
                  <Box pt="0.03125em" />
                ) : null}
                <GeneratedInput conf={conf} key={conf.id} viewer={viewer} />
                {conf.type === "GuiAddFolderMessage" &&
                index < guiIdSet.size - 1 ? (
                  // Add some whitespace after folders.
                  <Box pt="0.03125em" />
                ) : null}
              </>
            );
          })}
      </Box>
    );

  return out;
}

/** A single generated GUI element. */
function GeneratedInput({
  conf,
  viewer,
}: {
  conf: GuiConfig;
  viewer?: ViewerContextContents;
}) {
  // Handle GUI input types.
  if (viewer === undefined) viewer = React.useContext(ViewerContext)!;

  // Handle nested containers.
  if (conf.type == "GuiAddFolderMessage")
    return <GeneratedFolder conf={conf} />;
  if (conf.type == "GuiAddTabGroupMessage")
    return <GeneratedTabGroup conf={conf} />;
  if (conf.type == "GuiAddMarkdownMessage") {
    let { visible } =
      viewer.useGui((state) => state.guiAttributeFromId[conf.id]) || {};
    visible = visible ?? true;
    if (!visible) return <></>;
    return (
      <Box pb="xs" px="sm">
        <ErrorBoundary
          fallback={<Text align="center">Markdown Failed to Render</Text>}
        >
          <Markdown>{conf.markdown}</Markdown>
        </ErrorBoundary>
      </Box>
    );
  }

  const messageSender = makeThrottledMessageSender(viewer.websocketRef, 50);
  function updateValue(value: any) {
    setGuiValue(conf.id, value);
    messageSender({ type: "GuiUpdateMessage", id: conf.id, value: value });
  }

  const setGuiValue = viewer.useGui((state) => state.setGuiValue);
  const value =
    viewer.useGui((state) => state.guiValueFromId[conf.id]) ??
    conf.initial_value;
  const theme = useMantineTheme();

  let { visible, disabled } =
    viewer.useGui((state) => state.guiAttributeFromId[conf.id]) || {};

  visible = visible ?? true;
  disabled = disabled ?? false;

  if (!visible) return <></>;

  let inputColor =
    new ColorTranslator(theme.fn.primaryColor()).L > 55.0
      ? theme.colors.gray[9]
      : theme.white;

  let labeled = true;
  let input = null;
  switch (conf.type) {
    case "GuiAddButtonMessage":
      labeled = false;
      if (conf.color !== null) {
        inputColor =
          new ColorTranslator(theme.colors[conf.color][theme.fn.primaryShade()])
            .L > 55.0
            ? theme.colors.gray[9]
            : theme.white;
      }

      input = (
        <Button
          id={conf.id}
          fullWidth
          color={conf.color ?? undefined}
          onClick={() =>
            messageSender({
              type: "GuiUpdateMessage",
              id: conf.id,
              value: true,
            })
          }
          style={{ height: "1.875rem" }}
          styles={{ inner: { color: inputColor + " !important" } }}
          disabled={disabled}
          size="sm"
          leftIcon={
            conf.icon_base64 === null ? undefined : (
              <Image
                /*^In Safari, both the icon's height and width need to be set, otherwise the icon is clipped.*/
                height={"0.9rem"}
                width={"0.9rem"}
                opacity={disabled ? 0.3 : 1.0}
                sx={
                  inputColor === theme.white
                    ? {
                        // Make the color white.
                        filter: !disabled ? "invert(1)" : undefined,
                      }
                    : // Icon will be black by default.
                      undefined
                }
                src={"data:image/svg+xml;base64," + conf.icon_base64}
              />
            )
          }
        >
          {conf.label}
        </Button>
      );
      break;
    case "GuiAddSliderMessage":
      input = (
        <Flex justify="space-between">
          <Box sx={{ flexGrow: 1 }}>
            <Slider
              id={conf.id}
              size="sm"
              pt="0.3rem"
              showLabelOnHover={false}
              min={conf.min}
              max={conf.max}
              step={conf.step ?? undefined}
              precision={conf.precision}
              value={value}
              onChange={updateValue}
              marks={[{ value: conf.min }, { value: conf.max }]}
              disabled={disabled}
            />
            <Flex justify="space-between" sx={{ marginTop: "-0.2em" }}>
              <Text fz="0.7rem" c="dimmed">
                {parseInt(conf.min.toFixed(6))}
              </Text>
              <Text fz="0.7rem" c="dimmed">
                {parseInt(conf.max.toFixed(6))}
              </Text>
            </Flex>
          </Box>
          <NumberInput
            value={value}
            onChange={(newValue) => {
              // Ignore empty values.
              newValue !== "" && updateValue(newValue);
            }}
            size="xs"
            min={conf.min}
            max={conf.max}
            hideControls
            step={conf.step ?? undefined}
            precision={conf.precision}
            sx={{ width: "3rem", height: "1rem", minHeight: "1rem" }}
            styles={{ input: { padding: "0.3rem" } }}
            ml="xs"
          />
        </Flex>
      );
      break;
    case "GuiAddNumberMessage":
      input = (
        <NumberInput
          id={conf.id}
          value={value ?? conf.initial_value}
          precision={conf.precision}
          min={conf.min ?? undefined}
          max={conf.max ?? undefined}
          step={conf.step}
          size="xs"
          onChange={(newValue) => {
            // Ignore empty values.
            newValue !== "" && updateValue(newValue);
          }}
          disabled={disabled}
          stepHoldDelay={500}
          stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
        />
      );
      break;
    case "GuiAddTextMessage":
      input = (
        <TextInput
          value={value ?? conf.initial_value}
          size="xs"
          onChange={(value) => {
            updateValue(value.target.value);
          }}
          disabled={disabled}
        />
      );
      break;
    case "GuiAddCheckboxMessage":
      input = (
        <Checkbox
          id={conf.id}
          checked={value ?? conf.initial_value}
          size="xs"
          onChange={(value) => {
            updateValue(value.target.checked);
          }}
          disabled={disabled}
          styles={{
            icon: {
              color: inputColor + " !important",
            },
          }}
        />
      );
      break;
    case "GuiAddVector2Message":
      input = (
        <VectorInput
          id={conf.id}
          n={2}
          value={value ?? conf.initial_value}
          onChange={updateValue}
          min={conf.min}
          max={conf.max}
          step={conf.step}
          precision={conf.precision}
          disabled={disabled}
        />
      );
      break;
    case "GuiAddVector3Message":
      input = (
        <VectorInput
          id={conf.id}
          n={3}
          value={value ?? conf.initial_value}
          onChange={updateValue}
          min={conf.min}
          max={conf.max}
          step={conf.step}
          precision={conf.precision}
          disabled={disabled}
        />
      );
      break;
    case "GuiAddDropdownMessage":
      input = (
        <Select
          id={conf.id}
          value={value}
          data={conf.options}
          onChange={updateValue}
          searchable
          maxDropdownHeight={400}
          // zIndex of dropdown should be >modal zIndex.
          // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
          zIndex={1000}
          withinPortal={true}
        />
      );
      break;
    case "GuiAddRgbMessage":
      input = (
        <ColorInput
          disabled={disabled}
          size="xs"
          value={rgbToHex(value)}
          onChange={(v) => updateValue(hexToRgb(v))}
          format="hex"
          // zIndex of dropdown should be >modal zIndex.
          // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
          dropdownZIndex={1000}
          withinPortal={true}
        />
      );
      break;
    case "GuiAddRgbaMessage":
      input = (
        <ColorInput
          disabled={disabled}
          size="xs"
          value={rgbaToHex(value)}
          onChange={(v) => updateValue(hexToRgba(v))}
          format="hexa"
          // zIndex of dropdown should be >modal zIndex.
          // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
          dropdownZIndex={1000}
          withinPortal={true}
        />
      );
      break;
    case "GuiAddButtonGroupMessage":
      input = (
        <Flex justify="space-between" columnGap="xs">
          {conf.options.map((option, index) => (
            <Button
              key={index}
              onClick={() =>
                messageSender({
                  type: "GuiUpdateMessage",
                  id: conf.id,
                  value: option,
                })
              }
              style={{ flexGrow: 1, width: 0 }}
              disabled={disabled}
              compact
              size="sm"
              variant="outline"
            >
              {option}
            </Button>
          ))}
        </Flex>
      );
  }

  if (conf.hint !== null)
    input = // We need to add <Box /> for inputs that we can't assign refs to.
      (
        <Tooltip
          zIndex={100}
          label={conf.hint}
          multiline
          w="15rem"
          withArrow
          openDelay={500}
          withinPortal
        >
          <Box>{input}</Box>
        </Tooltip>
      );

  if (labeled)
    input = <LabeledInput id={conf.id} label={conf.label} input={input} />;

  return (
    <Box pb="xs" px="md">
      {input}
    </Box>
  );
}

function GeneratedFolder({ conf }: { conf: GuiAddFolderMessage }) {
  const [opened, { toggle }] = useDisclosure(true);
  const ToggleIcon = opened ? IconChevronUp : IconChevronDown;
  return (
    <Paper
      withBorder
      pt="0.2em"
      mx="xs"
      mt="sm"
      mb="sm"
      sx={{ position: "relative" }}
    >
      <Paper
        sx={{
          fontSize: "0.9em",
          position: "absolute",
          padding: "0 0.5em 0 0.25em",
          top: 0,
          left: "0.375em",
          transform: "translateY(-50%)",
          cursor: "pointer",
          userSelect: "none",
        }}
        onClick={toggle}
      >
        <ToggleIcon
          style={{
            width: "1.2em",
            height: "1.2em",
            top: "0.2em",
            position: "relative",
            marginRight: "0.25em",
            opacity: 0.5,
          }}
        />
        {conf.label}
      </Paper>
      <Collapse in={opened}>
        <GeneratedGuiContainer containerId={conf.id} />
      </Collapse>
      <Collapse in={!opened}>
        <Box p="xs"></Box>
      </Collapse>
    </Paper>
  );
}

function GeneratedTabGroup({ conf }: { conf: GuiAddTabGroupMessage }) {
  const [tabState, setTabState] = React.useState<TabsValue>("0");
  const icons = conf.tab_icons_base64;

  return (
    <Tabs
      radius="xs"
      value={tabState}
      onTabChange={setTabState}
      sx={(theme) => ({ marginTop: "-" + theme.spacing.xs })}
    >
      <Tabs.List>
        {conf.tab_labels.map((label, index) => (
          <Tabs.Tab
            value={index.toString()}
            key={index}
            icon={
              icons[index] === null ? undefined : (
                <Image
                  /*^In Safari, both the icon's height and width need to be set, otherwise the icon is clipped.*/
                  height={"0.9rem"}
                  width={"0.9rem"}
                  sx={(theme) => ({
                    filter:
                      theme.colorScheme == "dark" ? "invert(1)" : undefined,
                  })}
                  src={"data:image/svg+xml;base64," + icons[index]}
                />
              )
            }
          >
            {label}
          </Tabs.Tab>
        ))}
      </Tabs.List>
      {conf.tab_container_ids.map((containerId, index) => (
        <Tabs.Panel value={index.toString()} key={containerId}>
          <GeneratedGuiContainer containerId={containerId} />
        </Tabs.Panel>
      ))}
    </Tabs>
  );
}

function VectorInput(
  props:
    | {
        id: string;
        n: 2;
        value: [number, number];
        min: [number, number] | null;
        max: [number, number] | null;
        step: number;
        precision: number;
        onChange: (value: number[]) => void;
        disabled: boolean;
      }
    | {
        id: string;
        n: 3;
        value: [number, number, number];
        min: [number, number, number] | null;
        max: [number, number, number] | null;
        step: number;
        precision: number;
        onChange: (value: number[]) => void;
        disabled: boolean;
      },
) {
  return (
    <Flex justify="space-between" style={{ columnGap: "0.3rem" }}>
      {[...Array(props.n).keys()].map((i) => (
        <NumberInput
          id={i === 0 ? props.id : undefined}
          key={i}
          value={props.value[i]}
          onChange={(v) => {
            const updated = [...props.value];
            updated[i] = v === "" ? 0.0 : v;
            props.onChange(updated);
          }}
          size="xs"
          styles={{
            root: { flexGrow: 1, width: 0 },
            input: {
              paddingLeft: "0.3rem",
              paddingRight: "1.1rem",
              textAlign: "right",
            },
            rightSection: { width: "1.0rem" },
            control: {
              width: "0.875rem",
            },
          }}
          precision={props.precision}
          step={props.step}
          min={props.min === null ? undefined : props.min[i]}
          max={props.max === null ? undefined : props.max[i]}
          stepHoldDelay={500}
          stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
          disabled={props.disabled}
        />
      ))}
    </Flex>
  );
}

/** GUI input with a label horizontally placed to the left of it. */
function LabeledInput(props: {
  id: string;
  label: string;
  input: React.ReactNode;
}) {
  return (
    <Flex align="center">
      <Box w="6em" pr="xs">
        <Text
          c="dimmed"
          fz="sm"
          lh="1.15em"
          unselectable="off"
          sx={{ wordWrap: "break-word" }}
        >
          <label htmlFor={props.id}> {props.label}</label>
        </Text>
      </Box>
      <Box sx={{ flexGrow: 1 }}>{props.input}</Box>
    </Flex>
  );
}

// Color conversion helpers.

function rgbToHex([r, g, b]: [number, number, number]): string {
  const hexR = r.toString(16).padStart(2, "0");
  const hexG = g.toString(16).padStart(2, "0");
  const hexB = b.toString(16).padStart(2, "0");
  return `#${hexR}${hexG}${hexB}`;
}

function hexToRgb(hexColor: string): [number, number, number] {
  const hex = hexColor.slice(1); // Remove the # in #ffffff.
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return [r, g, b];
}
function rgbaToHex([r, g, b, a]: [number, number, number, number]): string {
  const hexR = r.toString(16).padStart(2, "0");
  const hexG = g.toString(16).padStart(2, "0");
  const hexB = b.toString(16).padStart(2, "0");
  const hexA = a.toString(16).padStart(2, "0");
  return `#${hexR}${hexG}${hexB}${hexA}`;
}

function hexToRgba(hexColor: string): [number, number, number, number] {
  const hex = hexColor.slice(1); // Remove the # in #ffffff.
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  const a = parseInt(hex.substring(6, 8), 16);
  return [r, g, b, a];
}
