// src/components/steps/Step1.jsx
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { FormField, FormItem, FormMessage } from '@/components/ui/form';
import { ageOptions, countryOptions, orgSizeOptions, employmentOptions } from '@/lib/data';

export const Step1 = ({ form }) => {
  const { register, control, formState: { errors } } = form;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Professional Background</CardTitle>
        <CardDescription>Let's start with your core experience.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="YearsCodePro">Years of Professional Coding</Label>
          <Input id="YearsCodePro" type="number" placeholder="e.g., 5" {...register('YearsCodePro')} />
          {errors.YearsCodePro && <p className="text-sm text-red-500 mt-1">{errors.YearsCodePro.message}</p>}
        </div>

        <div className="space-y-2">
          <Label htmlFor="WorkExp">Total Years of Work Experience</Label>
          <Input id="WorkExp" type="number" placeholder="e.g., 8" {...register('WorkExp')} />
          {errors.WorkExp && <p className="text-sm text-red-500 mt-1">{errors.WorkExp.message}</p>}
        </div>

        <div className="space-y-2">
          <Label htmlFor="CompTotal">Total Compensation (Local Currency)</Label>
          <Input id="CompTotal" type="number" placeholder="e.g., 60000" {...register('CompTotal')} />
          {errors.CompTotal && <p className="text-sm text-red-500 mt-1">{errors.CompTotal.message}</p>}
        </div>
        <div className="space-y-2">
          <Label>Age Range</Label>
          <FormField
            control={control}
            name="Age"
            render={({ field }) => (
              <FormItem>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select your age range">
                      {field.value || "Select your age range"}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {ageOptions.map(option => <SelectItem key={option} value={option}>{option}</SelectItem>)}
                  </SelectContent>
                </Select>
                {errors.Age && <FormMessage>{errors.Age.message}</FormMessage>}
              </FormItem>
            )}
          />
        </div>

        <div className="space-y-2">
          <Label>Country</Label>
          <FormField
            control={control}
            name="Country"
            render={({ field }) => (
              <FormItem>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select your country">
                      {field.value || "Select your country"}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {countryOptions.map(country => (
                      <SelectItem key={country} value={country}>{country}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {errors.Country && <FormMessage>{errors.Country.message}</FormMessage>}
              </FormItem>
            )}
          />
        </div>

        <div className="space-y-2">
          <Label>Organization Size</Label>
          <FormField
            control={control}
            name="OrgSize"
            render={({ field }) => (
              <FormItem>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select organization size">
                      {field.value || "Select organization size"}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {orgSizeOptions.map(option => <SelectItem key={option} value={option}>{option}</SelectItem>)}
                  </SelectContent>
                </Select>
                {errors.OrgSize && <FormMessage>{errors.OrgSize.message}</FormMessage>}
              </FormItem>
            )}
          />
        </div>

        <div className="space-y-2">
          <Label>Employment Status</Label>
          <FormField
            control={control}
            name="Employment"
            render={({ field }) => (
              <FormItem>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select employment status">
                      {field.value || "Select employment status"}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {employmentOptions.map(option => <SelectItem key={option} value={option}>{option}</SelectItem>)}
                  </SelectContent>
                </Select>
                {errors.Employment && <FormMessage>{errors.Employment.message}</FormMessage>}
              </FormItem>
            )}
          />
        </div>

      </CardContent>
    </Card>
  );
};