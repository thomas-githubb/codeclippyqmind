import React, { useState } from "react";
import axios from "axios";
import {
  Text,
  Box,
  Button,
  FormControl,
  FormLabel,
  Textarea,
  Container,
  VStack,
  Heading,
  useToast,
  Code,
  Skeleton,
} from "@chakra-ui/react";

function App() {
  const [loading, setLoading] = useState(false);
  const [generatedCode, setGeneratedCode] = useState("");
  const [isCodeGenerated, setIsCodeGenerated] = useState(false); // Track if code generation attempt was made
  const toast = useToast();

  function sendCode(event) {
    event.preventDefault();
    setLoading(true);
    setIsCodeGenerated(false); // Reset code generation attempt flag
    axios
      .get(`http://127.0.0.1:5000/generate_code_completion/${event.currentTarget.elements.code_input.value}`)
      .then((res) => {
        setGeneratedCode(res.data.generated_code);
        setIsCodeGenerated(true); // Set flag to indicate attempt was made
        toast({
          title: "Code generated successfully.",
          status: "success",
          duration: 5000,
          isClosable: true,
        });
      })
      .catch((error) => {
        console.error("There was an error!", error);
        setIsCodeGenerated(true); // Ensure user feedback even in case of error
        toast({
          title: "An error occurred.",
          description: "Unable to generate code.",
          status: "error",
          duration: 5000,
          isClosable: true,
        });
      })
      .finally(() => {
        setLoading(false);
      });
  }

  return (
    <Container maxW="container.md" centerContent p={5}>
      <VStack spacing={5} w="full">
        <Heading as="h1" size="xl">CodeClippy Code Generator</Heading>
        <form onSubmit={sendCode} style={{ width: "100%" }}>
          <FormControl>
            <FormLabel htmlFor="code_input">Code Fragment</FormLabel>
            <Textarea
              id="code_input"
              placeholder='def fib(x):'
              size='sm'
            />
            <Button mt={4} colorScheme="blue" isLoading={loading} type="submit" w="full">
              Submit
            </Button>
          </FormControl>
        </form>
        <Box w="full" p={4} bg="gray.100" mt={4} borderRadius="md" textAlign="left">
          {loading ? (
            <Box>
              <Skeleton height={'10em'}/>
            </Box>
          ) : isCodeGenerated ? (
            // Use pre and code tags to preserve formatting
            <pre><Code>{generatedCode}</Code></pre>
          ) : (
            <Text color="gray.500">Nothing Generated</Text>
          )}
        </Box>
      </VStack>
    </Container>
  );
}

export default App;
