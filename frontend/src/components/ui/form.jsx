import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import {
  Controller,
  useFormContext,
  useFormState,
  FormProvider as RHFFormProvider,
} from "react-hook-form";

import { cn } from "@/lib/utils";
import { Label } from "@/components/ui/label";

// Rename original FormProvider
const Form = RHFFormProvider;

// Contexts
const FormFieldContext = React.createContext(undefined);
const FormItemContext = React.createContext(undefined);

// Hook to access field metadata
const useFormField = () => {
  const fieldContext = React.useContext(FormFieldContext);
  const itemContext = React.useContext(FormItemContext);
  const formContext = useFormContext();

  if (!fieldContext || !fieldContext.name) {
    throw new Error("useFormField must be used inside a <FormField> with a 'name' prop.");
  }

  if (!formContext || !formContext.control) {
    throw new Error("useFormField must be used inside a <FormProvider>.");
  }

  const formState = useFormState({ name: fieldContext.name });
  const fieldState = formContext.getFieldState(fieldContext.name, formState);
  const id = itemContext?.id || fieldContext.name;

  return {
    id,
    name: fieldContext.name,
    formItemId: `${id}-form-item`,
    formDescriptionId: `${id}-form-item-description`,
    formMessageId: `${id}-form-item-message`,
    ...fieldState,
  };
};

// FormField component
const FormField = ({ name, render, ...props }) => {
  const methods = useFormContext();

  if (!methods || !methods.control) {
    throw new Error("FormField must be used inside a <FormProvider>.");
  }

  return (
    <FormFieldContext.Provider value={{ name }}>
      <Controller name={name} control={methods.control} render={render} {...props} />
    </FormFieldContext.Provider>
  );
};

// Wrapper for form items
const FormItem = ({ className, ...props }) => {
  const id = React.useId();

  return (
    <FormItemContext.Provider value={{ id }}>
      <div data-slot="form-item" className={cn("grid gap-2", className)} {...props} />
    </FormItemContext.Provider>
  );
};

// Label for a field
const FormLabel = ({ className, ...props }) => {
  const { error, formItemId } = useFormField();

  return (
    <Label
      data-slot="form-label"
      data-error={!!error}
      className={cn("data-[error=true]:text-destructive", className)}
      htmlFor={formItemId}
      {...props}
    />
  );
};

// Main control wrapper
const FormControl = (props) => {
  const { error, formItemId, formDescriptionId, formMessageId } = useFormField();

  return (
    <Slot
      data-slot="form-control"
      id={formItemId}
      aria-describedby={!error ? formDescriptionId : `${formDescriptionId} ${formMessageId}`}
      aria-invalid={!!error}
      {...props}
    />
  );
};

// Field description
const FormDescription = ({ className, ...props }) => {
  const { formDescriptionId } = useFormField();

  return (
    <p
      data-slot="form-description"
      id={formDescriptionId}
      className={cn("text-muted-foreground text-sm", className)}
      {...props}
    />
  );
};

// Field error message
const FormMessage = ({ className, children, ...props }) => {
  const { error, formMessageId } = useFormField();
  const content = error?.message || children;

  if (!content) return null;

  return (
    <p
      data-slot="form-message"
      id={formMessageId}
      className={cn("text-destructive text-sm", className)}
      {...props}
    >
      {String(content)}
    </p>
  );
};

// Final exports
export {
  Form,
  FormField,
  useFormField,
  FormItem,
  FormLabel,
  FormControl,
  FormDescription,
  FormMessage,
};