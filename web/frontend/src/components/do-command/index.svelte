<script lang="ts">
import { commonApi } from '@viamrobotics/sdk';
import { notify } from '@viamrobotics/prime';
import { resourceNameToString } from '@/lib/resource';
import { doCommand } from '@/api/do-command';
import Collapse from '@/lib/components/collapse.svelte';
import { useRobotClient } from '@/hooks/robot-client';
import { getClientByType } from './get-client-by-type';
import { Button, Label, SearchableSelect } from '@viamrobotics/prime-core';
import { isUnimplementedError } from './errors';

export let resources: commonApi.ResourceName.AsObject[];

const { robotClient } = useRobotClient();

let selectedComponent = undefined as
  | commonApi.ResourceName.AsObject
  | undefined;
let input = '{}';
let output = '';
let executing = false;

const handleDoCommand = async (
  type: string | undefined,
  name: string | undefined,
  command: string
) => {
  if (!type || !name || !command) {
    return;
  }

  const client = getClientByType($robotClient, type);
  if (!client) {
    return;
  }

  executing = true;

  try {
    const outputObject = await doCommand(client, name, command);

    if (outputObject) {
      output = JSON.stringify(outputObject, null, '\t');
      notify.success('Command issued');
    } else {
      notify.danger(`Invalid response when executing command on ${name}`);
    }
  } catch (error) {
    // Use a human-readable error message when we detect the do command is unimplemented
    if (isUnimplementedError(error)) {
      notify.danger(`DoCommand unimplemented for ${name}`);
    } else {
      notify.danger(
        `Error executing DoCommand on ${name}: ${JSON.stringify(error)}`
      );
    }
  } finally {
    executing = false;
  }
};

const handleSelectComponent = (value: string) => {
  selectedComponent = resources.find(({ name }) => name === value);
};

const handleEditorInput = (event: CustomEvent<{ value: string }>) => {
  input = event.detail.value;
};

let options: string[] = [];
$: {
  const simple = new Map<string, number>();

  for (const resource of resources) {
    if (!simple.has(resource.name)) {
      simple.set(resource.name, 0);
    }
    simple.set(resource.name, simple.get(resource.name)! + 1);
  }

  options = resources.map((resource) => {
    if (simple.get(resource.name) === 1) {
      return resource.name;
    }
    return resourceNameToString(resource);
  });
}
</script>

<Collapse title="DoCommand()">
  <div class="h-full w-full border border-t-0 border-medium p-4">
    <Label required>
      Selected component
      <SearchableSelect
        slot="input"
        {options}
        placeholder="Select a component"
        disabled={executing}
        exclusive
        onChange={handleSelectComponent}
      />
    </Label>
    <div class="flex h-full w-full flex-row flex-wrap gap-2">
      <div class="h-full w-full">
        <p class="text-sm">Input</p>
        <div class="h-[250px] w-full max-w-full border border-medium p-2">
          <v-code-editor
            language="json"
            value={'{}'}
            on:input={handleEditorInput}
          />
        </div>
      </div>
      <div class="flex min-w-[90px] flex-col justify-center">
        <Button
          disabled={!selectedComponent || !input || executing}
          on:click={async () =>
            handleDoCommand(
              selectedComponent?.subtype,
              selectedComponent?.name,
              input
            )}
        >
          Execute
        </Button>
      </div>
      <div class="h-full w-full">
        <p class="text-sm">Output</p>
        <div class="h-[250px] w-full border border-medium p-2">
          <v-code-editor
            language="json"
            value={output}
            readonly="true"
          />
        </div>
      </div>
    </div>
  </div>
</Collapse>
