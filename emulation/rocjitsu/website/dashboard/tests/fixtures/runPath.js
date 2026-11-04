// Independent test expectation; do not import the production path helper.
export const fixtureRunPath = ({ publication: { id, source } }) => `runs/${source.branch === 'develop' ? 'default-branch' : 'side-branches'}/${id}.json`;
