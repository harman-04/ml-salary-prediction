// src/components/steps/Step3.jsx
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'; // Import Select components
import { FormField, FormItem, FormControl, FormLabel, FormMessage } from '@/components/ui/form'; // Add FormMessage for error display
import {
  techSkills,
  knowledgeOptions,   // Import new options
  frequencyOptions,   // Import new options
  jobSatPointsOptions // Import new options
} from '@/lib/data';

export const Step3 = ({ form }) => {
  const { control } = form;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Skills & Work Habits</CardTitle> {/* Updated title */}
        <CardDescription>
          Select the technologies you have worked with and provide insights into your professional habits and knowledge.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Existing Tech Stack Accordion */}
        <Accordion type="multiple" className="w-full mb-8"> {/* Added mb-8 for spacing */}
          {Object.entries(techSkills).map(([category, skills]) => (
            <AccordionItem value={category} key={category}>
              <AccordionTrigger>{category}</AccordionTrigger>
              <AccordionContent>
                <FormField
                  control={control}
                  name={`${category}HaveWorkedWith`}
                  render={({ field }) => (
                    <FormItem className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                      {skills.map((skill) => {
                        const currentSelections = Array.isArray(field.value) ? field.value : [];
                        const isChecked = currentSelections.includes(skill);

                        return (
                          <FormItem key={skill} className="flex flex-row items-start space-x-3 space-y-0">
                            <FormControl>
                              <Checkbox
                                checked={isChecked}
                                onCheckedChange={(checked) => {
                                  const newValue = checked
                                    ? [...currentSelections, skill]
                                    : currentSelections.filter((v) => v !== skill);
                                  field.onChange(newValue);
                                }}
                              />
                            </FormControl>
                            <FormLabel className="font-normal">{skill}</FormLabel>
                          </FormItem>
                        );
                      })}
                    </FormItem>
                  )}
                />
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>

        {/* New Fields for Knowledge, Frequency, JobSatPoints */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Knowledge_1 */}
          <FormField
            control={control}
            name="Knowledge_1"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Importance of Cloud Computing knowledge:</FormLabel>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select importance" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    {knowledgeOptions.map((option) => (
                      <SelectItem key={option} value={option}>{option}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Knowledge_2 */}
          <FormField
            control={control}
            name="Knowledge_2"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Importance of Web Development knowledge:</FormLabel>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select importance" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    {knowledgeOptions.map((option) => (
                      <SelectItem key={option} value={option}>{option}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Frequency_1 */}
          <FormField
            control={control}
            name="Frequency_1"
            render={({ field }) => (
              <FormItem>
                <FormLabel>How often do you drive strategy for your team?</FormLabel>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select frequency" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    {frequencyOptions.map((option) => (
                      <SelectItem key={option} value={option}>{option}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* JobSatPoints_1 */}
          <FormField
            control={control}
            name="JobSatPoints_1"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Your satisfaction with career growth opportunities:</FormLabel>
                <Select onValueChange={field.onChange} defaultValue={field.value}>
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select satisfaction" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    {jobSatPointsOptions.map((option) => (
                      <SelectItem key={option} value={option}>{option}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
      </CardContent>
    </Card>
  );
};