// src/components/steps/Step2.jsx
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { FormField, FormItem, FormMessage } from '@/components/ui/form';
import { edLevelOptions, devTypeOptions, remoteWorkOptions, jobSatOptions } from '@/lib/data';

export const Step2 = ({ form }) => {
  const { control, formState: { errors } } = form;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Role, Education & Work Details</CardTitle>
        <CardDescription>Describe your current role, highest education, and work preferences.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label>Highest Education Level</Label>
          <FormField
            control={control}
            name="EdLevel"
            render={({ field }) => (
              <FormItem>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select education level">
                      {field.value || "Select education level"}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {edLevelOptions.map(option => <SelectItem key={option} value={option}>{option.split('(')[0]}</SelectItem>)}
                  </SelectContent>
                </Select>
                {errors.EdLevel && <FormMessage>{errors.EdLevel.message}</FormMessage>}
              </FormItem>
            )}
          />
        </div>
        <div className="space-y-2">
          <Label>Primary Role (Developer Type)</Label>
          <FormField
            control={control}
            name="DevType"
            render={({ field }) => (
              <FormItem>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select your primary role">
                      {field.value || "Select your primary role"}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {devTypeOptions.map(option => <SelectItem key={option} value={option}>{option}</SelectItem>)}
                  </SelectContent>
                </Select>
                {errors.DevType && <FormMessage>{errors.DevType.message}</FormMessage>}
              </FormItem>
            )}
          />
        </div>

        <div className="space-y-2">
          <Label>Remote Work Preference</Label>
          <FormField
            control={control}
            name="RemoteWork"
            render={({ field }) => (
              <FormItem>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select your remote work preference">
                      {field.value || "Select your remote work preference"}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {remoteWorkOptions.map(option => <SelectItem key={option} value={option}>{option}</SelectItem>)}
                  </SelectContent>
                </Select>
                {errors.RemoteWork && <FormMessage>{errors.RemoteWork.message}</FormMessage>}
              </FormItem>
            )}
          />
        </div>

        <div className="space-y-2">
          <Label>Job Satisfaction</Label>
          <FormField
            control={control}
            name="JobSat"
            render={({ field }) => (
              <FormItem>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select your job satisfaction">
                      {field.value || "Select your job satisfaction"}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {jobSatOptions.map(option => <SelectItem key={option} value={option}>{option}</SelectItem>)}
                  </SelectContent>
                </Select>
                {errors.JobSat && <FormMessage>{errors.JobSat.message}</FormMessage>}
              </FormItem>
            )}
          />
        </div>

      </CardContent>
    </Card>
  );
};